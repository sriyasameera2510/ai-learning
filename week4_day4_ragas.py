from groq import Groq
from sentence_transformers import SentenceTransformer
from langsmith import traceable
import chromadb
import json
import numpy as np
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "clinical-ai-assistant"

# === SETUP ===
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db_v2")
collection = chroma_client.get_or_create_collection(name="patient_records")


# === RAG PIPELINE ===
def retrieve_chunks(question: str) -> list:
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    return results['documents'][0]


def generate_answer(question: str, chunks: list) -> str:
    context = "\n\n".join(chunks)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "Answer using ONLY the provided records. Be specific. If not found say so."
            },
            {
                "role": "user",
                "content": f"Records:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content


# === RAGAS-STYLE METRICS ===

@traceable(name="faithfulness_score")
def measure_faithfulness(answer: str, chunks: list) -> float:
    """
    Faithfulness — does the answer stick to what the context says?
    Method: ask LLM to check each claim in the answer against the context
    Score: 0.0 to 1.0
    """
    context = "\n\n".join(chunks)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": f"""Given this context and answer, score faithfulness from 0.0 to 1.0.
Faithfulness = are all claims in the answer supported by the context?
1.0 = every claim is in the context
0.5 = some claims unsupported
0.0 = answer contradicts or ignores context

CONTEXT: {context[:800]}
ANSWER: {answer}

Respond with JSON only: {{"score": 0.0, "reason": "brief reason"}}"""
        }]
    )
    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1].replace("json", "").strip()
    try:
        return json.loads(raw)["score"]
    except Exception:
        return 0.5


@traceable(name="answer_relevancy_score")
def measure_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy — does the answer address the question?
    Method: embed both, measure cosine similarity
    Score: 0.0 to 1.0
    """
    q_embedding = embed_model.encode([question])[0]
    a_embedding = embed_model.encode([answer])[0]

    # Cosine similarity
    similarity = np.dot(q_embedding, a_embedding) / (
        np.linalg.norm(q_embedding) * np.linalg.norm(a_embedding)
    )
    return round(float(similarity), 3)


@traceable(name="context_precision_score")
def measure_context_precision(question: str, chunks: list) -> float:
    """
    Context Precision — are retrieved chunks actually relevant?
    Method: ask LLM to rate each chunk's relevance to the question
    Score: proportion of chunks that are relevant
    """
    relevant = 0
    for chunk in chunks:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": f"""Is this chunk relevant to answering the question?
QUESTION: {question}
CHUNK: {chunk[:300]}

Respond with JSON only: {{"relevant": true or false}}"""
            }]
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        try:
            if json.loads(raw)["relevant"]:
                relevant += 1
        except Exception:
            relevant += 0.5

    return round(relevant / len(chunks), 3)


@traceable(name="context_recall_score")
def measure_context_recall(
    question: str,
    chunks: list,
    ground_truth: str
) -> float:
    """
    Context Recall — did we retrieve everything needed to answer?
    Method: check if ground truth information is present in retrieved chunks
    Score: 0.0 to 1.0
    """
    context = "\n\n".join(chunks)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": f"""Does the retrieved context contain enough information 
to produce this ground truth answer?

QUESTION: {question}
GROUND TRUTH: {ground_truth}
RETRIEVED CONTEXT: {context[:800]}

Score from 0.0 to 1.0:
1.0 = context contains everything needed
0.5 = context has partial information
0.0 = context is missing key information

Respond with JSON only: {{"score": 0.0, "reason": "brief reason"}}"""
        }]
    )
    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1].replace("json", "").strip()
    try:
        return json.loads(raw)["score"]
    except Exception:
        return 0.5


# === TEST CASES ===
test_cases = [
    {
        "question": "What are Jane D's current CA-125 levels?",
        "ground_truth": "Jane D's current CA-125 is 340, down from 520 three months ago."
    },
    {
        "question": "What medications is Robert M currently on?",
        "ground_truth": "Robert M is on pembrolizumab and prescribed blood thinners, though non-adherent due to cost."
    },
    {
        "question": "Is Sarah K currently pregnant?",
        "ground_truth": "Yes, Sarah K is 8 weeks pregnant which impacts her cancer treatment."
    },
    {
        "question": "Does any patient have diabetes?",
        "ground_truth": "None of the patient records mention diabetes."
    },
    {
        "question": "What is Robert M's CEA marker trend?",
        "ground_truth": "Robert M's CEA markers increased from 4.2 to 11.8 over two months."
    }
]


# === RUN EVALUATION ===
@traceable(name="ragas_style_eval_suite", tags=["evaluation"])
def run_ragas_style_evals():
    print("\n" + "="*60)
    print("RAGAS-STYLE EVALUATION SUITE")
    print(f"Running {len(test_cases)} test cases")
    print("="*60)

    all_scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": []
    }

    for tc in test_cases:
        print(f"\n{'─'*60}")
        print(f"Q: {tc['question']}")

        # Run RAG
        chunks = retrieve_chunks(tc["question"])
        answer = generate_answer(tc["question"], chunks)
        print(f"A: {answer[:100]}...")

        # Measure all 4 metrics
        faith = measure_faithfulness(answer, chunks)
        relevancy = measure_answer_relevancy(tc["question"], answer)
        precision = measure_context_precision(tc["question"], chunks)
        recall = measure_context_recall(
            tc["question"], chunks, tc["ground_truth"]
        )

        all_scores["faithfulness"].append(faith)
        all_scores["answer_relevancy"].append(relevancy)
        all_scores["context_precision"].append(precision)
        all_scores["context_recall"].append(recall)

        print(f"\n  Faithfulness:      {faith:.3f}")
        print(f"  Answer Relevancy:  {relevancy:.3f}")
        print(f"  Context Precision: {precision:.3f}")
        print(f"  Context Recall:    {recall:.3f}")

    # Averages
    print("\n" + "="*60)
    print("AVERAGE SCORES")
    print("="*60)

    overall = []
    for metric, scores in all_scores.items():
        avg = sum(scores) / len(scores)
        overall.append(avg)
        print(f"{metric.ljust(22)}: {avg:.3f}")

    print(f"\nOverall RAG Score:     {sum(overall)/len(overall):.3f}")

    # Save
    import csv
    with open("ragas_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question", "faithfulness", "answer_relevancy",
            "context_precision", "context_recall"
        ])
        for i, tc in enumerate(test_cases):
            writer.writerow([
                tc["question"],
                all_scores["faithfulness"][i],
                all_scores["answer_relevancy"][i],
                all_scores["context_precision"][i],
                all_scores["context_recall"][i]
            ])

    print("\nResults saved to ragas_results.csv")


run_ragas_style_evals()