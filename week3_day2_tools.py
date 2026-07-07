from groq import Groq
from dotenv import load_dotenv
from ddgs import DDGS
import os
import math

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# === REAL TOOLS ===

def calculator(expression: str) -> str:
    """
    Safely evaluates a math expression.
    Examples: "340/520*100", "math.sqrt(144)", "(520-340)/520*100"
    """
    try:
        # Safe eval — only allows math operations
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {round(result, 4)}"
    except Exception as e:
        return f"Error: {str(e)}"


def web_search(query: str) -> str:
    """
    Searches the web and returns top 3 results.
    Use for current medical guidelines, drug information, clinical data.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if not results:
            return "No results found"

        output = []
        for r in results:
            output.append(f"Title: {r['title']}\nSummary: {r['body']}\n")

        return "\n".join(output)
    except Exception as e:
        return f"Search error: {str(e)}"


def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Converts between medical units.
    Supports: mg/mcg/g, ml/l, mmol/mol
    """
    conversions = {
        ("mg", "mcg"): 1000,
        ("mcg", "mg"): 0.001,
        ("g", "mg"): 1000,
        ("mg", "g"): 0.001,
        ("l", "ml"): 1000,
        ("ml", "l"): 0.001,
        ("mmol", "mol"): 0.001,
        ("mol", "mmol"): 1000,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = value * conversions[key]
        return f"{value} {from_unit} = {result} {to_unit}"
    return f"Conversion from {from_unit} to {to_unit} not supported"


# === TOOL REGISTRY ===
tools = {
    "calculator": calculator,
    "web_search": web_search,
    "unit_converter": unit_converter
}

tool_descriptions = """
Available tools:
1. calculator(expression)
   - Evaluates math expressions
   - Examples: calculator("(520-340)/520*100"), calculator("math.sqrt(144)")

2. web_search(query)
   - Searches the web for current information
   - Use for drug info, medical guidelines, clinical data
   - Example: web_search("carboplatin paclitaxel side effects ovarian cancer")

3. unit_converter(value, from_unit, to_unit)
   - Converts medical units
   - Supported: mg/mcg/g, ml/l, mmol/mol
   - Example: unit_converter(500, "mg", "mcg")
"""


# === REACT AGENT (same loop as yesterday) ===
def run_agent(question: str, max_steps: int = 8):
    print(f"\n{'=' * 60}")
    print(f"QUESTION: {question}")
    print('=' * 60)

    messages = [
        {
            "role": "system",
            "content": f"""You are a clinical AI assistant with access to real tools.

{tool_descriptions}

STRICT RULES:
- Only ONE action per response, then stop and wait
- Always use correct Python syntax in tool calls
- Never assume tool results — wait for the OBSERVATION
- Use web_search when you need current medical information

Respond in EXACTLY this format:
THOUGHT: your reasoning
ACTION: tool_name(params)

When done:
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

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            messages=messages
        )

        agent_output = response.choices[0].message.content
        print(agent_output)

        if "FINAL ANSWER:" in agent_output:
            print("\n✅ Agent completed task")
            break

        if "ACTION:" in agent_output:
            action_line = [l for l in agent_output.split('\n') if l.startswith('ACTION:')][0]
            action = action_line.replace('ACTION:', '').strip()

            try:
                tool_result = eval(action, {"__builtins__": {}}, tools)
                observation = f"OBSERVATION: {tool_result}"
            except Exception as e:
                observation = f"OBSERVATION: Error - {str(e)}"

            print(observation)

            messages.append({"role": "assistant", "content": agent_output})
            messages.append({"role": "user", "content": observation})


# === TEST WITH REAL QUESTIONS ===
run_agent(
    "A patient is prescribed 250mg of a drug twice daily. Convert the daily dose to mcg, then search for the maximum safe daily dose of carboplatin."
)