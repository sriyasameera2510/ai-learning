from groq import Groq
from sentence_transformers import SentenceTransformer
from langsmith import traceable, Client
import chromadb
import math
import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

# === LANGSMITH SETUP ===
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "clinical-ai-assistant"

ls_client = Client()
print("🔍 LangSmith tracing enabled")
print(f"📊 Project: clinical-ai-assistant\n")

# === SETUP ===
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db_v2")
collection = chroma_client.get_or_create_collection(name="patient_records")


# === TOOLS ===
@traceable(name="lookup_tool", tags=["tool", "retrieval"])
def lookup(query: str) -> str:
    """Look up patient records"""
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    docs = results['documents'][0]
    sources = [meta['source_file'] for meta in results['metadatas'][0]]
    output = [f"[{src}]: {doc}" for doc, src in zip(docs, sources)]
    return "\n\n".join(output)


@traceable(name="calculator_tool", tags=["tool", "math"])
def calculator(expression: str) -> str:
    """Evaluate math expression"""
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{round(result, 4)}"
    except Exception as e:
        return f"Error: {str(e)}"


# === MANUAL REACT AGENT WITH TRACING ===
# Going back to our Week 3 Day 1 approach — more reliable than langchain-groq

tools_map = {
    "lookup": lookup,
    "calculator": calculator
}

tool_descriptions = """
Available tools:
1. lookup("query") - search patient medical records
   Example: lookup("Jane D CA-125 readings")

2. calculator("expression") - evaluate math with numbers only
   Example: calculator("(520-340)/520*100")
"""


@traceable(name="clinical_agent", tags=["agent", "clinical"])
def run_query(question: str):
    print(f"\n{'─'*60}")
    print(f"Query: {question}")
    print('─'*60)

    messages = [
        {
            "role": "system",
            "content": f"""You are a Clinical AI Assistant.
{tool_descriptions}

STRICT RULES:
- ONE action per response then stop
- Use quoted strings in tool calls: lookup("jane d ca-125")
- Never nest tool calls
- Stop after ACTION line and wait for observation

Format:
THOUGHT: your reasoning
ACTION: tool_name("param")

When done:
THOUGHT: I have enough information
FINAL ANSWER: your answer"""
        },
        {
            "role": "user",
            "content": question
        }
    ]

    for step in range(6):
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            messages=messages
        )

        agent_output = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        if "FINAL ANSWER:" in agent_output:
            final = agent_output.split("FINAL ANSWER:")[-1].strip()
            print(f"\n💬 {final}")
            print(f"📊 Tokens used this step: {tokens_used}")
            break

        if "ACTION:" in agent_output:
            print(f"\n  Step {step+1}:")
            print(f"  {[l for l in agent_output.split(chr(10)) if 'THOUGHT:' in l][0] if 'THOUGHT:' in agent_output else ''}")

            action_line = [l for l in agent_output.split('\n') if l.startswith('ACTION:')][0]
            action = action_line.replace('ACTION:', '').strip()

            try:
                tool_result = eval(action, {"__builtins__": {}}, tools_map)
                observation = f"OBSERVATION: {tool_result}"
                print(f"  🔧 {action}")
                print(f"  📥 {str(tool_result)[:100]}...")
            except Exception as e:
                observation = f"OBSERVATION: Error - {str(e)}"
                print(f"  ⚠️ Tool error: {str(e)}")

            messages.append({"role": "assistant", "content": agent_output})
            messages.append({"role": "user", "content": observation})

    return agent_output


# === RUN TRACED QUERIES ===
run_query("What are Jane D's CA-125 readings?")
run_query("What medications is Robert M on?")
run_query("Calculate the percentage change in CEA markers from 4.2 to 11.8")

print("\n✅ Check smith.langchain.com → clinical-ai-assistant project")