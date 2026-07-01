import os
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db_v2")
collection = chroma_client.get_or_create_collection(
    name="patient_records",
    metadata={"hnsw:space": "cosine"}
)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def retrieve_with_filter(query: str, source_file: str = None, n_results: int = 4):
    """Retrieve with optional metadata filter"""
    query_embedding = embed_model.encode([query]).tolist()

    # Only filter if a source file is specified
    where_filter = {"source_file": source_file} if source_file else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where_filter  # ← this is metadata filtering
    )

    docs = results['documents'][0]
    sources = [meta['source_file'] for meta in results['metadatas'][0]]
    distances = results['distances'][0]

    return docs, sources, distances


# Test 1 — without filter (old behavior)
print("=" * 60)
print("WITHOUT FILTER")
print("=" * 60)
docs, sources, distances = retrieve_with_filter(
    "Is treatment showing positive progress",
)
for doc, source, dist in zip(docs, sources, distances):
    print(f"\n[{source}] similarity: {1 - dist:.3f}")
    print(f"{doc[:100]}...")

# Test 2 — with filter (only search Jane D.'s file)
print("\n\n" + "=" * 60)
print("WITH METADATA FILTER (only patient_1.txt)")
print("=" * 60)
docs, sources, distances = retrieve_with_filter(
    "Is treatment showing positive progress",
    source_file="patient_1.txt"
)
for doc, source, dist in zip(docs, sources, distances):
    print(f"\n[{source}] similarity: {1 - dist:.3f}")
    print(f"{doc[:100]}...")

    # Load a cross-encoder reranker
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


    def retrieve_and_rerank(query: str, source_file: str = None, n_retrieve: int = 6, n_final: int = 3):
        """
        Two-stage retrieval:
        Stage 1 — embedding similarity (fast, rough)
        Stage 2 — cross-encoder reranking (slow, precise)
        """

        # Stage 1: retrieve more candidates than you need
        query_embedding = embed_model.encode([query]).tolist()
        where_filter = {"source_file": source_file} if source_file else None

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_retrieve,  # grab 6 candidates
            where=where_filter
        )

        docs = results['documents'][0]
        sources = [meta['source_file'] for meta in results['metadatas'][0]]

        print(f"\nStage 1 retrieved {len(docs)} candidates")

        # Stage 2: rerank using cross-encoder
        # CrossEncoder scores each (query, document) pair together
        pairs = [[query, doc] for doc in docs]
        rerank_scores = reranker.predict(pairs)

        # Sort by rerank score (highest = most relevant)
        ranked = sorted(
            zip(rerank_scores, docs, sources),
            reverse=True
        )

        print(f"Stage 2 reranked — returning top {n_final}\n")

        # Return only the top N after reranking
        top_docs = [doc for _, doc, _ in ranked[:n_final]]
        top_sources = [source for _, _, source in ranked[:n_final]]
        top_scores = [score for score, _, _ in ranked[:n_final]]

        return top_docs, top_sources, top_scores


    # Compare old vs new on the failing question
    query = "Is treatment showing positive progress"

    print("\n" + "=" * 60)
    print("BEFORE RERANKING (embedding only)")
    print("=" * 60)
    docs, sources, distances = retrieve_with_filter(query, source_file="patient_1.txt")
    for doc, source, dist in zip(docs, sources, distances):
        print(f"\nEmbedding score: {1 - dist:.3f} [{source}]")
        print(f"{doc[:120]}...")

    print("\n" + "=" * 60)
    print("AFTER RERANKING (embedding + cross-encoder)")
    print("=" * 60)
    docs, sources, scores = retrieve_and_rerank(query, source_file="patient_1.txt")
    for doc, source, score in zip(docs, sources, scores):
        print(f"\nRerank score: {score:.3f} [{source}]")
        print(f"{doc[:120]}...")