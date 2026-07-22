from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# === SETUP ===
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db_v2")
collection = chroma_client.get_or_create_collection(name="patient_records")


# === YOUR RAG PIPELINE FROM WEEK 2 ===
def rag_query(question: str) -> str:
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    docs = results['documents'][0]
    sources = [meta['source_file'] for meta in results['metadatas'][0]]
    context = "\n\n".join([f"[{src}]: {doc}" for doc, src in zip(docs, sources)])

    response = client.chat.completions.create(
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


# === EVAL DATASET ===
# Each test case has:
# - question: what you ask the system
# - expected_keywords: words that MUST appear in a correct answer
# - forbidden_keywords: words that should NOT appear (hallucination check)
# - source_file: which file should be cited
# - notes: why this test case exists

eval_dataset = [
    {
        "id": "eval_001",
        "question": "What are Jane D's current CA-125 levels?",
        "expected_keywords": ["340", "520", "patient_1"],
        "forbidden_keywords": ["11.8", "4.2"],  # Robert M's markers
        "source_file": "patient_1.txt",
        "notes": "Tests basic retrieval and correct patient attribution"
    },
    {
        "id": "eval_002",
        "question": "What medications is Robert M currently on?",
        "expected_keywords": ["pembrolizumab", "patient_2"],
        "forbidden_keywords": ["carboplatin", "paclitaxel"],  # Jane D's meds
        "source_file": "patient_2.txt",
        "notes": "Tests medication retrieval without cross-patient contamination"
    },
    {
        "id": "eval_003",
        "question": "Is Sarah K currently pregnant?",
        "expected_keywords": ["pregnant", "8 weeks", "patient_3"],
        "forbidden_keywords": [],
        "source_file": "patient_3.txt",
        "notes": "Tests retrieval of critical patient safety information"
    },
    {
        "id": "eval_004",
        "question": "What is Jane D's blood pressure reading?",
        "expected_keywords": ["145", "90", "patient_1"],
        "forbidden_keywords": ["Robert", "Sarah"],
        "source_file": "patient_1.txt",
        "notes": "Tests specific numeric retrieval"
    },
    {
        "id": "eval_005",
        "question": "Does any patient have diabetes?",
        "expected_keywords": ["no", "not", "don't", "no mention", "cannot find"],
        "forbidden_keywords": ["yes", "diabetic", "insulin"],
        "source_file": None,
        "notes": "Hallucination test — correct answer is that no data exists"
    },
    {
        "id": "eval_006",
        "question": "What is Robert M's CEA marker trend?",
        "expected_keywords": ["4.2", "11.8", "patient_2"],
        "forbidden_keywords": ["CA-125", "340", "520"],  # Jane D's markers
        "source_file": "patient_2.txt",
        "notes": "Tests marker retrieval and correct marker type attribution"
    }
]


# === EVALUATION RUNNER ===
def evaluate_response(response: str, test_case: dict) -> dict:
    """Score a response against expected criteria"""
    response_lower = response.lower()

    # Check expected keywords
    keyword_results = {}
    for keyword in test_case["expected_keywords"]:
        keyword_results[keyword] = keyword.lower() in response_lower

    # Check forbidden keywords (hallucination detection)
    forbidden_results = {}
    for keyword in test_case["forbidden_keywords"]:
        forbidden_results[keyword] = keyword.lower() in response_lower

    # Calculate scores
    keyword_score = sum(keyword_results.values()) / len(keyword_results) if keyword_results else 1.0
    hallucination_detected = any(forbidden_results.values())

    # Overall pass/fail
    passed = keyword_score >= 0.7 and not hallucination_detected

    return {
        "passed": passed,
        "keyword_score": round(keyword_score, 2),
        "hallucination_detected": hallucination_detected,
        "keyword_results": keyword_results,
        "forbidden_results": forbidden_results
    }


def run_evals():
    """Run the full evaluation suite"""
    print("\n" + "="*60)
    print("RAG SYSTEM EVALUATION")
    print(f"Running {len(eval_dataset)} test cases")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

    results = []
    passed = 0
    failed = 0

    for test_case in eval_dataset:
        print(f"\n[{test_case['id']}] {test_case['question']}")
        print(f"Notes: {test_case['notes']}")

        # Get system response
        response = rag_query(test_case["question"])

        # Evaluate it
        eval_result = evaluate_response(response, test_case)

        # Track results
        status = "✅ PASS" if eval_result["passed"] else "❌ FAIL"
        if eval_result["passed"]:
            passed += 1
        else:
            failed += 1

        print(f"Status: {status}")
        print(f"Keyword score: {eval_result['keyword_score']}")
        print(f"Hallucination detected: {eval_result['hallucination_detected']}")

        if not eval_result["passed"]:
            print(f"Response: {response[:200]}...")
            missing = [k for k, v in eval_result["keyword_results"].items() if not v]
            if missing:
                print(f"Missing keywords: {missing}")
            hallucinated = [k for k, v in eval_result["forbidden_results"].items() if v]
            if hallucinated:
                print(f"Hallucinated terms: {hallucinated}")

        results.append({
            "test_case": test_case,
            "response": response,
            "eval_result": eval_result
        })

    # Final summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total:  {len(eval_dataset)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Score:  {passed/len(eval_dataset)*100:.1f}%")

    # Save results
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to eval_results.json")

    return results


# === RUN IT ===
run_evals()