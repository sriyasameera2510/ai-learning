from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# temperature controlshow creative/predictable the output is
# 0.0 – 0.2 : Extracting data, classification, structured output — you want the same answer every time
# 0.5 – 0.7 : Most apps, chatbots, summaries — balanced and reliable
# 1.0+ : Brainstorming, creative writing, name generation — you want surprises

prompt = "Give me 3 creative names for an AI-powered oncology assistant"

for temp in [0.0, 0.7, 1.5]:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"\n--- Temperature: {temp} ---")
    print(response.choices[0].message.content)