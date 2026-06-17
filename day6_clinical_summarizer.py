from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def summarize_with_stats(note: str, patient_num: int):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": """You are a clinical data extraction assistant for an oncology platform.
Extract information from clinical notes and return ONLY this exact structure, nothing else:

PATIENT SUMMARY: One sentence describing the patient
DIAGNOSIS: Primary diagnosis and stage if available
MEDICATIONS: Comma separated list of current medications
ACTION ITEMS: Bullet points of what the oncologist should follow up on
RISK LEVEL: Low / Medium / High with one sentence reason

Think step by step before extracting each field."""
            },
            {
                "role": "user",
                "content": f"Extract from this clinical note:\n\n{note}"
            }
        ]
    )

    result = response.choices[0].message.content
    tokens = response.usage.total_tokens
    cost = tokens * 0.000001

    return result, tokens, cost


# Test with 3 different notes
notes = [
    """
    Patient Jane D., 54F, presented today for follow-up after her second cycle of 
    chemotherapy for stage III ovarian cancer. She reports increased fatigue and 
    some nausea but manageable. CA-125 markers came back at 340, down from 520 
    last visit which is encouraging. Currently on carboplatin and paclitaxel. 
    Blood pressure slightly elevated at 145/90. Next imaging scheduled in 6 weeks. 
    Patient expressed anxiety about upcoming scan results.
    """,
    """
    Robert M., 67M, long term patient with stage II non-small cell lung cancer. 
    Today's visit shows concerning rise in CEA markers from 4.2 to 11.8 over 
    past 2 months. Currently on pembrolizumab. Patient reports new onset of 
    persistent cough and occasional chest pain. Has not been taking prescribed 
    blood thinners regularly. Referred for urgent CT scan. Family history of 
    blood clots noted.
    """,
    """
    Sarah K., 41F, newly diagnosed with HER2 positive breast cancer stage I. 
    Patient is in good overall health. No current medications. Discussed 
    treatment options including surgery followed by trastuzumab. Patient is 
    pregnant — 8 weeks. Treatment plan needs oncology and obstetrics 
    collaboration. Patient is anxious but cooperative.
    """
]

for i, note in enumerate(notes, 1):
    result, tokens, cost = summarize_with_stats(note, i)
    print(f"\n{'='*50}")
    print(f"PATIENT {i}")
    print(f"Tokens used: {tokens} | Cost estimate: ${cost:.6f}")
    print('='*50)
    print(result)