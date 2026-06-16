from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

question = """
A cancer patient is on 3 medications. Each costs $240/month. 
Their insurance covers 60% of medication costs. 
They also have a $50/month copay per medication. 
What is their total monthly out-of-pocket cost?
"""

# Without CoT
response_basic = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    messages=[
        {"role": "user", "content": question}
    ]
)

# With CoT
response_cot = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": question + "\nThink through this step by step before giving the final answer."
        }
    ]
)

print("--- WITHOUT Chain-of-Thought ---")
print(response_basic.choices[0].message.content)

print("\n--- WITH Chain-of-Thought ---")
print(response_cot.choices[0].message.content)