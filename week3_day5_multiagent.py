from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
import chromadb
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv
import os
import math
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# === RAG SETUP ===
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db_v2")
collection = chroma_client.get_or_create_collection(name="patient_records")
print(f"ChromaDB loaded — {collection.count()} chunks\n")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# === SHARED TOOLS ===

@tool
def search_records(query: str) -> str:
    """Search patient medical records for relevant information."""
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    docs = results['documents'][0]
    sources = [meta['source_file'] for meta in results['metadatas'][0]]
    output = [f"[{source}]: {doc}" for doc, source in zip(docs, sources)]
    return "\n\n".join(output)


@tool
def calculator(expression: str) -> str:
    """Safely evaluates math expressions. Pass numbers only."""
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {round(result, 4)}"
    except Exception as e:
        return f"Error: {str(e)}"


# === AGENT 1 — RESEARCHER ===
researcher = create_react_agent(
    model=llm,
    tools=[search_records, calculator],
    prompt="""You are a clinical research assistant. Your job is to:
1. Search patient records thoroughly
2. Extract all relevant clinical data
3. Calculate any relevant metrics
4. Present findings clearly with source citations

Be thorough — find ALL relevant information before concluding."""
)


# === AGENT 2 — CRITIC ===
critic = create_react_agent(
    model=llm,
    tools=[search_records],
    prompt="""You are a senior clinical reviewer. Your job is to:
1. Review the researcher's findings critically
2. Identify any gaps, missing information, or potential errors
3. Search for any contradicting or additional evidence
4. Flag any clinical concerns the researcher may have missed
5. Rate the completeness of the research: LOW / MEDIUM / HIGH

Be skeptical and thorough — your job is to find what was missed."""
)


# === STATE ===
class MultiAgentState(TypedDict):
    question: str
    research_findings: str
    critic_review: str
    final_answer: str
    messages: Annotated[list, operator.add]


# === NODES ===

def researcher_node(state: MultiAgentState):
    """Researcher agent does the initial investigation"""
    print("\n" + "="*60)
    print("🔍 RESEARCHER AGENT")
    print("="*60)

    result = researcher.invoke({
        "messages": [HumanMessage(content=f"Research this clinical question thoroughly: {state['question']}")]
    })

    # Extract final answer from researcher
    findings = ""
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  [Tool] {tc['name']}({tc['args']})")
        elif type(msg).__name__ == "AIMessage" and msg.content:
            findings = msg.content
            print(f"\n  [Findings] {findings}")

    return {
        "research_findings": findings,
        "messages": [AIMessage(content=f"RESEARCHER: {findings}")]
    }


def critic_node(state: MultiAgentState):
    """Critic agent reviews and challenges the research"""
    print("\n" + "="*60)
    print("🔎 CRITIC AGENT")
    print("="*60)

    critic_prompt = f"""Review these research findings about: {state['question']}

RESEARCHER'S FINDINGS:
{state['research_findings']}

Critically review this. Search for anything missing or concerning."""

    result = critic.invoke({
        "messages": [HumanMessage(content=critic_prompt)]
    })

    review = ""
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  [Tool] {tc['name']}({tc['args']})")
        elif type(msg).__name__ == "AIMessage" and msg.content:
            review = msg.content
            print(f"\n  [Review] {review}")

    return {
        "critic_review": review,
        "messages": [AIMessage(content=f"CRITIC: {review}")]
    }


def synthesizer_node(state: MultiAgentState):
    """Final synthesis combining research and critique"""
    print("\n" + "="*60)
    print("📋 FINAL SYNTHESIS")
    print("="*60)

    synthesis_prompt = f"""You are a senior oncologist synthesizing a clinical report.

ORIGINAL QUESTION: {state['question']}

RESEARCHER FINDINGS:
{state['research_findings']}

CRITIC REVIEW:
{state['critic_review']}

Synthesize both into a final, comprehensive clinical summary.
Include: key findings, any concerns raised by the critic, and recommended actions."""

    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
    final = response.content
    print(f"\n{final}")

    return {
        "final_answer": final,
        "messages": [AIMessage(content=f"FINAL: {final}")]
    }


# === BUILD THE GRAPH ===
graph = StateGraph(MultiAgentState)

graph.add_node("researcher", researcher_node)
graph.add_node("critic", critic_node)
graph.add_node("synthesizer", synthesizer_node)

graph.set_entry_point("researcher")
graph.add_edge("researcher", "critic")
graph.add_edge("critic", "synthesizer")
graph.add_edge("synthesizer", END)

pipeline = graph.compile()


# === RUN IT ===
def run_multi_agent(question: str):
    print(f"\n{'='*60}")
    print(f"CLINICAL QUESTION: {question}")
    print("="*60)

    result = pipeline.invoke({
        "question": question,
        "research_findings": "",
        "critic_review": "",
        "final_answer": "",
        "messages": []
    })

    return result["final_answer"]


run_multi_agent(
    "Assess Robert M's current clinical status. "
    "Are his tumor markers and symptoms concerning? "
    "What should the oncologist prioritize?"
)