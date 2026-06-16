from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    messages=[
        {
            "role": "system",
            "content": """You are a clinical data analyst assistant.
When answering any question, always follow this structure:
THINKING: Walk through your reasoning step by step
ANSWER: Give your final concise answer
CONFIDENCE: Rate your confidence as Low / Medium / High and explain why"""
        },
        {
            "role": "user",
            "content": "A patient's tumor markers have increased 3 months in a row by 15%, 22%, and 18%. Is this clinically significant and what would you flag for the oncologist?"
        }
    ]
)

print(response.choices[0].message.content)