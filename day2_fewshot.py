from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Few shot prompting is to teach AI by example

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    messages=[
        {
            "role": "system",
            "content": "You extract structured data from clinical notes. Return only in this exact format:\nPatient Age: X\nDiagnosis: X\nMedication: X"
        },
        # Example 1 — you show it input
        {
            "role": "user",
            "content": "Patient is a 45-year-old female diagnosed with breast cancer stage 2. Currently on tamoxifen."
        },
        # Example 1 — you show it the expected output
        {
            "role": "assistant",
            "content": "Patient Age: 45\nDiagnosis: Breast cancer stage 2\nMedication: Tamoxifen"
        },
        # Example 2
        {
            "role": "user",
            "content": "68 year old male, lung carcinoma detected last month. Prescribed cisplatin and pemetrexed."
        },
        {
            "role": "assistant",
            "content": "Patient Age: 68\nDiagnosis: Lung carcinoma\nMedication: Cisplatin, Pemetrexed"
        },
        # Now the real input — no example given, AI follows the pattern
        {
            "role": "user",
            "content": "We have a 52 year old woman who came in with ovarian cancer. She has been started on carboplatin."
        }
    ]
)

print(response.choices[0].message.content)