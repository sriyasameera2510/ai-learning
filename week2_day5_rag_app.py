import os
import chromadb
from sentence_transformers import SentenceTransformer
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


def chunk_by_sentence(text: str, max_chunk_size: int = 300):
    """Sentence-aware chunking from Day 4"""
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def ingest_folder(folder_path: str):
    """Read all .txt files, chunk them, embed them, store in ChromaDB"""

    all_chunks = []
    all_ids = []
    all_metadata = []

    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    print(f"Found {len(files)} patient files to ingest\n")

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r') as f:
            text = f.read()

        chunks = chunk_by_sentence(text)
        print(f"  {filename} -> {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}_chunk_{i}")
            all_metadata.append({"source_file": filename, "chunk_index": i})

    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    embeddings = embed_model.encode(all_chunks).tolist()

    collection.upsert(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metadata
    )

    print(f"Ingestion complete. Collection now has {collection.count()} chunks.\n")


def retrieve(query: str, n_results: int = 4):
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    # Return both the text AND which file it came from
    docs = results['documents'][0]
    sources = [meta['source_file'] for meta in results['metadatas'][0]]
    return docs, sources


def generate_answer(query: str, docs: list, sources: list):
    context = "\n\n".join([
        f"[From {source}]: {doc}"
        for doc, source in zip(docs, sources)
    ])

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": """You are a clinical assistant. Answer questions using ONLY the provided patient record excerpts.
Always cite which patient file your information came from.
If the answer isn't in the provided excerpts, say "I don't have that information in the records." """
            },
            {
                "role": "user",
                "content": f"Patient Record Excerpts:\n{context}\n\nQuestion: {query}"
            }
        ]
    )
    return response.choices[0].message.content


def ask(query: str):
    print(f"\n{'=' * 60}")
    print(f"QUESTION: {query}")
    print('=' * 60)

    docs, sources = retrieve(query)
    print(f"\n[Retrieved {len(docs)} chunks from: {set(sources)}]")

    answer = generate_answer(query, docs, sources)
    print(f"\nANSWER:\n{answer}")


# === RUN THE PIPELINE ===
ingest_folder("patient_records")

ask("Which patient has a complication with pregnancy?")
ask("What medication concerns has Robert M. had?")
ask("Is Jane D.'s treatment showing positive progress?")
ask("Does any patient have diabetes?")