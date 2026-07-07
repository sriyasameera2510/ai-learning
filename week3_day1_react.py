from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# === DEFINE TOOLS ===
# These are just Python functions — the agent will decide when to call them

def search_patient_records(patient_name: str, field: str) -> str:
    """Simulated patient record search"""
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


def calculate_percentage_change(old_value: float, new_value: float) -> str:
    """Calculate percentage change between two values"""
    change = ((new_value - old_value) / old_value) * 100
    direction = "decrease" if change < 0 else "increase"
    return f"{abs(change):.1f}% {direction}"


def check_drug_interaction(drug1: str, drug2: str) -> str:
    """Simulated drug interaction checker"""
    interactions = {
        ("carboplatin",
         "paclitaxel"): "Known combination — standard ovarian cancer regimen, monitor for peripheral neuropathy",
        ("pembrolizumab", "blood thinners"): "Caution — immunotherapy can increase bleeding risk with anticoagulants"
    }

    key = tuple(sorted([drug1.lower(), drug2.lower()]))
    return interactions.get(key, f"No known interaction data for {drug1} and {drug2}")


# === TOOL REGISTRY ===
# Maps tool names to functions so the agent can call them by name
tools = {
    "search_patient_records": search_patient_records,
    "calculate_percentage_change": calculate_percentage_change,
    "check_drug_interaction": check_drug_interaction
}

tool_descriptions = """
Available tools:
1. search_patient_records(patient_name, field)
   - patient_name options: "jane d", "robert m"
   - field options: "ca-125", "medications", "diagnosis", 
                    "blood pressure", "cea markers", "symptoms"

2. calculate_percentage_change(old_value, new_value)
   - Pass numbers NOT strings: calculate_percentage_change(520, 340)

3. check_drug_interaction(drug1, drug2)
   - Example: check_drug_interaction("carboplatin", "paclitaxel")
"""


# === THE REACT AGENT LOOP ===

def run_react_agent(question: str, max_steps: int = 6):
    print(f"\n{'=' * 60}")
    print(f"QUESTION: {question}")
    print('=' * 60)

    messages = [
        {
            "role": "system",
            "content": f"""You are a clinical AI assistant. Solve problems step by step using available tools.

{tool_descriptions}

STRICT RULES:
- Only ONE action per response, then stop and wait
- Always use quoted strings in tool calls exactly like this:
  search_patient_records("jane d", "ca-125")
- Never combine multiple actions in one response
- Never assume tool results — wait for the OBSERVATION

For each step respond in EXACTLY this format:
THOUGHT: your reasoning
ACTION: tool_name("param1", "param2")

Stop after the ACTION line and wait for the observation.

When you have enough information:
THOUGHT: I now have enough information
FINAL ANSWER: your complete answer"""
        },
        {
            "role": "user",
            "content": question
        }
    ]

    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")

        # Get agent's next thought and action
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            messages=messages
        )

        agent_output = response.choices[0].message.content
        print(agent_output)

        # Check if agent is done
        if "FINAL ANSWER:" in agent_output:
            print("\n✅ Agent completed task")
            break

        # Parse and execute the tool call
        if "ACTION:" in agent_output:
            action_line = [line for line in agent_output.split('\n') if line.startswith('ACTION:')][0]
            action = action_line.replace('ACTION:', '').strip()

            try:
                # Execute the tool — eval is safe here since we control the tool registry
                tool_result = eval(action, {"__builtins__": {}}, tools)
                observation = f"OBSERVATION: {tool_result}"
            except Exception as e:
                observation = f"OBSERVATION: Error executing tool - {str(e)}"

            print(observation)

            # Add both agent output and observation to history
            messages.append({"role": "assistant", "content": agent_output})
            messages.append({"role": "user", "content": observation})


# === TEST THE AGENT ===
run_react_agent(
    "Is Jane D.'s CA-125 trend showing improvement? Calculate the exact percentage change and tell me if her current medications are appropriate together."
)