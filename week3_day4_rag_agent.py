from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq as GroqDirect
from dotenv import load_dotenv
import os
import math
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

# === SETUP RAG COMPONENTS ===
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db_v2")
collection = chroma_client.get_or_create_collection(name="patient_records")
groq_direct = GroqDirect(api_key=os.getenv("GROQ_API_KEY"))

# Quick connection test
print(f"ChromaDB loaded — {collection.count()} chunks available")

# Test a quick query
test_embedding = embed_model.encode(["test"]).tolist()
test_result = collection.query(query_embeddings=test_embedding, n_results=1)
print(f"Test query successful: {test_result['documents'][0][0][:50]}")


# === TOOLS ===

@tool
def rag_search(query: str) -> str:
    """Search patient medical records and return relevant information.
    Use this for any question about patient history, diagnoses,
    medications, symptoms, or clinical notes.
    Example: rag_search('Sarah K treatment plan')
    """
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    docs = results['documents'][0]
    sources = [meta['source_file'] for meta in results['metadatas'][0]]

    # Return raw chunks — the agent LLM will synthesize the answer
    output = []
    for doc, source in zip(docs, sources):
        output.append(f"[{source}]: {doc}")

    return "\n\n".join(output)


@tool
def calculator(expression: str) -> str:
    """Safely evaluates a math expression.
    Example: calculator('(520-340)/520*100')
    Only pass numbers and math operators — never nest other tool calls.
    """
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {round(result, 4)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def flag_urgent(patient_name: str, reason: str) -> str:
    """Flag a patient as requiring urgent clinical attention.
    Use when you identify critical findings like rapidly rising markers,
    medication non-adherence, or dangerous symptoms.
    Example: flag_urgent('Robert M', 'CEA markers tripled in 2 months')
    """
    # In production this would write to a database or send an alert
    alert = f"🚨 URGENT FLAG — Patient: {patient_name} | Reason: {reason}"
    print(f"\n{alert}\n")
    return f"Urgent flag created for {patient_name}: {reason}"


# === SETUP AGENT ===
tools = [rag_search, calculator, flag_urgent]
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="""You are a senior clinical AI assistant with access to patient records.

RULES:
- Use rag_search for ANY question about patient information
- Use calculator for math — but ONLY with actual numbers, never nest tool calls
- Use flag_urgent when you identify critical clinical findings
- Always cite which patient file your information came from
- One tool at a time"""
)


# === RUN IT ===
def run_agent(question: str):
    print(f"\n{'=' * 60}")
    print(f"QUESTION: {question}")
    print('=' * 60)

    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    for msg in result["messages"]:
        msg_type = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\n[TOOL CALL] {tc['name']}({tc['args']})")
        elif hasattr(msg, "name") and msg.name:
            print(f"\n[TOOL RESULT] {msg.content[:200]}...")
        elif msg_type == "AIMessage" and msg.content:
            print(f"\n[AI] {msg.content}")


# === TEST QUESTIONS ===

# Test 1 — pure RAG retrieval
run_agent("What is the current treatment plan for Sarah K.?")

# Test 2 — RAG + calculation
run_agent(
    "Look up Robert M's CEA markers from the records, "
    "then calculate the percentage increase from 4.2 to 11.8"
)

# Test 3 — RAG + urgent flag (tests agent decision making)
run_agent(
    "Review Robert M's latest symptoms and marker trends. "
    "If anything looks clinically concerning, flag it as urgent."
)