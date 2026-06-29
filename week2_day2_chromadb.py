import chromadb
from sentence_transformers import SentenceTransformer

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB — stores data locally in a folder
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection — think of this like a database table
collection = client.get_or_create_collection(
    name="clinical_notes",
    metadata={"hnsw:space": "cosine"}  # use cosine similarity
)

# Our clinical documents — simulating a real oncology database
documents = [
    "Jane D., 54F, stage III ovarian cancer. CA-125 down from 520 to 340. On carboplatin and paclitaxel. Blood pressure elevated.",
    "Robert M., 67M, stage II non-small cell lung cancer. CEA markers rising from 4.2 to 11.8. On pembrolizumab. Urgent CT scan ordered.",
    "Sarah K., 41F, HER2 positive breast cancer stage I. Pregnant 8 weeks. No current medications. Needs oncology and obstetrics collaboration.",
    "Michael T., 58M, colorectal cancer stage II. Post surgery recovery. Starting adjuvant FOLFOX chemotherapy next week. Good overall health.",
    "Linda R., 63F, chronic lymphocytic leukemia. Stable disease. On ibrutinib. Last CBC showed improving white cell counts.",
    "David W., 71M, prostate cancer stage III. PSA rising from 4.1 to 12.3. Referred for radiation oncology consultation. On leuprolide.",
    "Emma P., 45F, thyroid cancer post thyroidectomy. TSH suppression therapy with levothyroxine. No evidence of recurrence on last scan.",
    "James H., 55M, pancreatic cancer stage IV. On gemcitabine and nab-paclitaxel. Palliative care consultation recommended.",
]

# Generate embeddings for all documents
print("Embedding documents...")
embeddings = model.encode(documents).tolist()

# Store in ChromaDB with IDs and metadata
collection.upsert(
    documents=documents,
    embeddings=embeddings,
    ids=[f"patient_{i}" for i in range(len(documents))],
    metadatas=[{"patient_num": i, "indexed": "2024"} for i in range(len(documents))]
)

print(f"Stored {collection.count()} documents in ChromaDB\n")

# Now search with natural language queries
queries = [
    "patient with rising tumor markers needing urgent imaging",
    "pregnant patient with cancer requiring special care",
    "patient with stable disease on targeted therapy",
    "patient with poor prognosis needing palliative care"
]

for query in queries:
    print(f"\nQuery: '{query}'")
    print("-" * 50)

    # Embed the query
    query_embedding = model.encode([query]).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2  # return top 2 matches
    )

    for i, (doc, distance) in enumerate(zip(
        results['documents'][0],
        results['distances'][0]
    )):
        print(f"  Match {i+1} (similarity: {1-distance:.3f}): {doc[:80]}...")