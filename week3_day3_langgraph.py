from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
import math
import warnings
warnings.filterwarnings("ignore")  # suppress deprecation warnings

load_dotenv()


# === DEFINE TOOLS ===
@tool
def calculator(expression: str) -> str:
    """Safely evaluates a math expression.
    Example: calculator('(520-340)/520*100')
    Always pass numbers directly, never nest other tool calls inside.
    """
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {round(result, 4)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_patient(patient_name: str, field: str) -> str:
    """Search patient records.
    patient_name options: 'jane d', 'robert m'
    field options: 'ca-125', 'medications', 'diagnosis',
                   'blood pressure', 'cea markers', 'symptoms'
    """
    records = {
        "jane d": {
            "ca-125": "Current: 340, Previous: 520 (3 months ago)",
            "medications": "Carboplatin, Paclitaxel",
            "diagnosis": "Stage III ovarian cancer",
            "blood pressure": "145/90 - slightly elevated"
        },
        "robert m": {
            "cea markers": "Current: 11.8, Previous: 4.2 (2 months ago)",
            "medications": "Pembrolizumab, Blood thinners (non-adherent)",
            "diagnosis": "Stage II non-small cell lung cancer",
            "symptoms": "Persistent cough, occasional chest pain"
        }
    }
    patient = patient_name.lower()
    field = field.lower()
    if patient in records and field in records[patient]:
        return records[patient][field]
    return f"No record found for {patient_name} - {field}"


# === SETUP ===
tools = [calculator, search_patient]
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="You are a clinical AI assistant. Use tools one at a time. Never nest tool calls. First retrieve data, then calculate separately."
)


# === RUN IT ===
def run_agent(question: str):
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print('='*60)

    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    for msg in result["messages"]:
        msg_type = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\n[TOOL CALL] {tc['name']}({tc['args']})")
        elif hasattr(msg, "name") and msg.name:
            print(f"\n[TOOL RESULT] {msg.content}")
        elif msg_type == "AIMessage" and msg.content:
            print(f"\n[AI] {msg.content}")


# Test 1 — single tool
run_agent("What is Jane D.'s diagnosis?")

# Test 2 — multiple tools
run_agent("What is Jane D.'s CA-125 readings and what is her diagnosis?")

# Test 3 — retrieval + calculation
run_agent("Search Jane D.'s CA-125 readings first, then calculate the percentage change between 520 and 340.")