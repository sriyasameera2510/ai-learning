from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# This list grows with every exchange — this IS the memory
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant for oncology data engineers. You give concise, technical answers."
    }
]

print("Chat started! Type 'quit' to exit.\n")

while True:
    # Get input from you
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Ending chat.")
        break

    # Append your message to history
    messages.append({
        "role": "user",
        "content": user_input
    })

    # Send full history to API
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        messages=messages
    )

    # Extract reply
    reply = response.choices[0].message.content

    # Append AI reply to history too
    messages.append({
        "role": "assistant",
        "content": reply
    })

    print(f"\nAI: {reply}\n")