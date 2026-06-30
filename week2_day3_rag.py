import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Setup
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="clinical_notes")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def retrieve(query: str, n_results: int = 3):
    """Step 1 & 2: Embed query and search ChromaDB"""
    query_embedding = embed_model.encode([query]).tolist()
    """give me the documents whose vectors are closest to this one"""
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results['documents'][0]


def generate_answer(query: str, retrieved_docs: list):
    """Step 3 & 4: Stuff context into prompt, ask LLM"""

    # Build the context block from retrieved documents
    context = "\n\n".join([f"Document {i + 1}: {doc}" for i, doc in enumerate(retrieved_docs)])

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": """You are a clinical assistant. Answer questions ONLY using the provided patient documents below.
If the answer isn't in the documents, say "I don't have that information in the records."
Always mention which patient(s) your answer is based on."""
            },
            {
                "role": "user",
                "content": f"""Patient Documents:
{context}

Question: {query}"""
            }
        ]
    )
    return response.choices[0].message.content


def rag_query(query: str):
    """The full RAG pipeline"""
    print(f"\n{'=' * 60}")
    print(f"QUESTION: {query}")
    print('=' * 60)

    # Retrieval
    docs = retrieve(query)
    print(f"\n[Retrieved {len(docs)} relevant documents]")

    # Generation
    answer = generate_answer(query, docs)
    print(f"\nANSWER:\n{answer}")


# Test it with real questions
rag_query("Which patients have rising cancer markers and need urgent follow-up?")
rag_query("Is there a patient with special considerations due to pregnancy?")
rag_query("What treatment is Linda R. currently on?")
rag_query("Does any patient have diabetes?")  # tests "I don't know" behavior