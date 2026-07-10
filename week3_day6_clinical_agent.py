from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from sentence_transformers import SentenceTransformer
from datetime import datetime
import chromadb
import math
import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# SETUP
# ============================================================
print("\n🏥 Clinical AI Assistant starting up...")

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db_v2")
collection = chroma_client.get_or_create_collection(name="patient_records")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

urgent_flags = []

print(f"✅ Loaded {collection.count()} patient record chunks")
print("✅ LLM connected")
print("✅ Ready\n")


# ============================================================
# TOOLS — simplified to 3 for Groq stability
# ============================================================

@tool
def search_records(query: str) -> str:
    """Search patient medical records for clinical information.
    Use for: diagnoses, medications, symptoms, lab results, history.
    Example: search_records('Jane D CA-125')
    """
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    docs = results['documents'][0]
    sources = [meta['source_file'] for meta in results['metadatas'][0]]
    output = [f"[{src}]: {doc}" for doc, src in zip(docs, sources)]
    return "\n\n".join(output)


@tool
def calculator(expression: str) -> str:
    """Evaluates math expressions safely.
    Use for: percentage changes, dose calculations.
    Example: calculator('(520-340)/520*100')
    Pass numbers only — never nest tool calls.
    """
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{round(result, 4)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def flag_urgent(patient_name: str, concern: str, priority: str) -> str:
    """Flag a patient for urgent clinical attention.
    priority options: 'HIGH' or 'CRITICAL'
    Use HIGH: significant marker changes, medication non-adherence with risk
    Use CRITICAL: immediate safety concerns, rapidly deteriorating markers
    Example: flag_urgent('Robert M', 'CEA tripled in 2 months', 'HIGH')
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    flag = {
        "time": timestamp,
        "patient": patient_name,
        "concern": concern,
        "priority": priority
    }
    urgent_flags.append(flag)

    alert = f"""
╔══════════════════════════════════════════╗
║  🚨 URGENT FLAG [{priority}]
║  Patient:  {patient_name}
║  Concern:  {concern[:50]}
║  Time:     {timestamp}
╚══════════════════════════════════════════╝"""
    print(alert)
    return f"Urgent flag created — {priority} priority for {patient_name}"


# ============================================================
# AGENT
# ============================================================

tools = [search_records, calculator, flag_urgent]

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="""You are a Clinical AI Assistant for oncology teams.

    TOOLS:
    - search_records: search patient files for any clinical information
    - calculator: do math with actual numbers only
    - flag_urgent: flag patients needing urgent attention (HIGH or CRITICAL)

    RULES:
    - Always search records before making clinical statements
    - Calculator takes numbers only — never nest tool calls inside calculator
    - Flag CRITICAL for immediate safety concerns
    - Flag HIGH for significant marker changes or medication non-adherence with risk
    - Always cite which source file your information came from
    - One tool at a time
    - IMPORTANT: Call flag_urgent ONLY ONCE per patient per assessment.
      Consolidate ALL concerns into a single flag_urgent call.""")


# ============================================================
# QUERY RUNNER
# ============================================================

def run_query(question: str):
    print(f"\n{'─'*60}")
    print(f"Query: {question}")
    print('─'*60)

    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=question)]
        })

        for msg in result["messages"]:
            msg_type = type(msg).__name__
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    args_str = str(tc['args'])[:80]
                    print(f"  🔧 {tc['name']}({args_str})")
            elif msg_type == "AIMessage" and msg.content:
                print(f"\n💬 {msg.content}")

    except Exception as e:
        print(f"\n⚠️ Agent error: {str(e)[:200]}")
        print("Try rephrasing your question.")


# ============================================================
# SESSION SUMMARY
# ============================================================

def show_session_summary():
    print(f"\n{'='*60}")
    if urgent_flags:
        print(f"SESSION SUMMARY — {len(urgent_flags)} urgent flag(s) raised")
        print('='*60)
        for flag in urgent_flags:
            print(f"  [{flag['priority']}] {flag['patient']}: {flag['concern']}")
    else:
        print("SESSION SUMMARY — No urgent flags this session ✅")
    print('='*60)


# ============================================================
# INTERACTIVE INTERFACE
# ============================================================

def main():
    print("="*60)
    print("🏥 CLINICAL AI ASSISTANT")
    print("="*60)
    print("Type your clinical questions below.")
    print("Commands: 'summary' = session flags | 'quit' = exit")
    print("="*60)

    while True:
        try:
            user_input = input("\n👨‍⚕️ Clinician: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                show_session_summary()
                print("\nGoodbye! 👋")
                break

            if user_input.lower() == 'summary':
                show_session_summary()
                continue

            run_query(user_input)

        except KeyboardInterrupt:
            show_session_summary()
            print("\nSession ended.")
            break


if __name__ == "__main__":
    main()