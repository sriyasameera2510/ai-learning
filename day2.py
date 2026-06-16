from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

 #System prompts control role,tone and scope simultaneously. They give AI a personality/role

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are a senior oncology data engineer. You explain technical concepts clearly and always relate them back to healthcare data use cases."
        },
        {
            "role": "user",
            "content": "What is an LLM and how could it be used in a cancer care platform?"
        }
    ]
)

print(response.choices[0].message.content)