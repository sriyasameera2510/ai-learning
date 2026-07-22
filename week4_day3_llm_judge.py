from groq import Groq
from sentence_transformers import SentenceTransformer
from langsmith import traceable
import chromadb
import json
import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "clinical-ai-assistant"

# === SETUP ===
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db_v2")
collection = chroma_client.get_or_create_collection(name="patient_records")


# === RAG PIPELINE ===
@traceable(name="rag_retrieve")
def retrieve(question: str) -> tuple:
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    docs = results['documents'][0]
    sources = [meta['source_file'] for meta in results['metadatas'][0]]
    context = "\n\n".join([f"[{src}]: {doc}" for doc, src in zip(docs, sources)])
    return context, sources


@traceable(name="rag_generate")
def generate(question: str, context: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "Answer using ONLY the provided patient records. Be specific and cite sources. If not found say so."
            },
            {
                "role": "user",
                "content": f"Records:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content


@traceable(name="rag_query")
def rag_query(question: str) -> tuple:
    context, sources = retrieve(question)
    answer = generate(question, context)
    return answer, context


# === LLM JUDGE ===
@traceable(name="llm_judge", tags=["evaluation", "judge"])
def llm_judge(question: str, context: str, answer: str) -> dict:
    """Use an LLM to evaluate the quality of a RAG answer"""

    judge_prompt = f"""You are an expert clinical AI evaluator. 
Evaluate the following RAG system response on 4 dimensions.

QUESTION: {question}

RETRIEVED CONTEXT:
{context[:1000]}

SYSTEM ANSWER: {answer}

Score each dimension and respond in valid JSON only — no other text:

{{
    "accuracy": {{
        "score": <1-5>,
        "reasoning": "<why this score>"
    }},
    "groundedness": {{
        "score": <1-5>,
        "reasoning": "<is the answer supported by the context or does it add unsupported claims>"
    }},
    "completeness": {{
        "score": <1-5>,
        "reasoning": "<does the answer fully address the question>"
    }},
    "hallucination": {{
        "detected": <true or false>,
        "reasoning": "<what was hallucinated, if anything>"
    }},
    "overall_score": <1-5>,
    "verdict": "<PASS or FAIL>",
    "feedback": "<one sentence summary of the main issue or strength>"
}}

Scoring guide:
5 = excellent, 4 = good, 3 = acceptable, 2 = poor, 1 = wrong
PASS = overall_score >= 3 and hallucination.detected = false"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        messages=[{"role": "user", "content": judge_prompt}]
    )

    raw = response.choices[0].message.content.strip()

    # Clean up JSON if wrapped in markdown
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw)
    except Exception:
        return {
            "error": "Judge failed to return valid JSON",
            "raw": raw,
            "verdict": "ERROR"
        }


# === EVAL DATASET ===
eval_dataset = [
    {
        "id": "eval_001",
        "question": "What are Jane D's current CA-125 levels?",
        "notes": "Basic retrieval and correct patient attribution"
    },
    {
        "id": "eval_002",
        "question": "What medications is Robert M currently on?",
        "notes": "Medication retrieval without cross-patient contamination"
    },
    {
        "id": "eval_003",
        "question": "Is Sarah K currently pregnant?",
        "notes": "Critical patient safety information retrieval"
    },
    {
        "id": "eval_004",
        "question": "What is Jane D's blood pressure reading?",
        "notes": "Specific numeric retrieval"
    },
    {
        "id": "eval_005",
        "question": "Does any patient have diabetes?",
        "notes": "Hallucination test — correct answer is no data exists"
    },
    {
        "id": "eval_006",
        "question": "What is Robert M's CEA marker trend?",
        "notes": "Marker retrieval and correct type attribution"
    }
]


# === RUN LLM-AS-JUDGE EVALS ===
@traceable(name="llm_judge_eval_suite", tags=["evaluation"])
def run_llm_judge_evals():
    print("\n" + "="*60)
    print("LLM-AS-JUDGE EVALUATION SUITE")
    print(f"Running {len(eval_dataset)} test cases")
    print("="*60)

    results = []
    passed = 0
    failed = 0
    errors = 0

    for test_case in eval_dataset:
        print(f"\n[{test_case['id']}] {test_case['question']}")
        print(f"Notes: {test_case['notes']}")

        # Get RAG answer
        answer, context = rag_query(test_case["question"])

        # Judge it
        judgment = llm_judge(test_case["question"], context, answer)

        if "error" in judgment:
            print(f"Status: ⚠️  JUDGE ERROR")
            errors += 1
        else:
            verdict = judgment.get("verdict", "ERROR")
            overall = judgment.get("overall_score", 0)
            hallucination = judgment.get("hallucination", {}).get("detected", False)
            feedback = judgment.get("feedback", "")

            status = "✅ PASS" if verdict == "PASS" else "❌ FAIL"
            if verdict == "PASS":
                passed += 1
            else:
                failed += 1

            print(f"Status:       {status}")
            print(f"Overall:      {overall}/5")
            print(f"Accuracy:     {judgment.get('accuracy', {}).get('score', 'N/A')}/5")
            print(f"Groundedness: {judgment.get('groundedness', {}).get('score', 'N/A')}/5")
            print(f"Completeness: {judgment.get('completeness', {}).get('score', 'N/A')}/5")
            print(f"Hallucination:{hallucination}")
            print(f"Feedback:     {feedback}")

        results.append({
            "test_case": test_case,
            "answer": answer,
            "judgment": judgment
        })

    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY — LLM-AS-JUDGE")
    print("="*60)
    print(f"Total:  {len(eval_dataset)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Errors: {errors} ⚠️")
    if passed + failed > 0:
        print(f"Score:  {passed/(passed+failed)*100:.1f}%")

    # Save
    with open("llm_judge_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to llm_judge_results.json")

    return results


# === RUN IT ===
run_llm_judge_evals()