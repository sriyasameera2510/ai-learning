from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Simulate a LONG document — a multi-section patient history
long_document = """
Patient Sarah K. Medical History.

Initial Presentation: Sarah K., 41 years old, presented in January 2024 with a palpable lump in the right breast. Mammogram revealed a suspicious mass measuring 2.3cm. Biopsy confirmed HER2 positive invasive ductal carcinoma, stage I.

Treatment Plan: Given the early stage diagnosis, the oncology team recommended a treatment approach combining surgery with targeted therapy. The patient was scheduled for lumpectomy followed by trastuzumab treatment over 12 months.

Pregnancy Complication: During pre-treatment workup, it was discovered that Sarah is currently 8 weeks pregnant. This significantly complicates the treatment timeline. The oncology team has initiated collaboration with the obstetrics department to determine safe treatment windows that protect both maternal and fetal health.

Current Status: As of the latest visit, Sarah remains in good overall health. No current medications have been started pending the joint oncology-obstetrics consultation. Patient reports feeling anxious but is cooperative and engaged in the decision making process.

Follow Up Plan: A joint consultation between oncology and maternal-fetal medicine specialists is scheduled for next week to finalize a treatment plan that accounts for the pregnancy.
"""


def chunk_naive(text: str, chunk_size: int = 200):
    """Bad approach — cuts mid-sentence, ignores structure"""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_by_sentence(text: str, max_chunk_size: int = 300):
    """Better approach — never cuts mid-sentence"""
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


def chunk_with_overlap(text: str, max_chunk_size: int = 300, overlap: int = 50):
    """Best approach — sentence aware AND preserves context across chunk boundaries"""
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk with the END of the previous chunk (the overlap)
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


print("\n\n" + "=" * 60)
print("OVERLAPPING CHUNKS (preserves context across boundaries)")
print("=" * 60)
overlap_chunks = chunk_with_overlap(long_document)
for i, chunk in enumerate(overlap_chunks):
    print(f"\nChunk {i + 1} ({len(chunk)} chars):")
    print(chunk)

# Compare both approaches
print("=" * 60)
print("NAIVE CHUNKING (cuts mid-sentence)")
print("=" * 60)
naive_chunks = chunk_naive(long_document)
for i, chunk in enumerate(naive_chunks[:3]):  # just show first 3
    print(f"\nChunk {i + 1}:")
    print(repr(chunk))  # repr shows exact cut points

print("\n\n" + "=" * 60)
print("SENTENCE-AWARE CHUNKING (respects sentence boundaries)")
print("=" * 60)
smart_chunks = chunk_by_sentence(long_document)
for i, chunk in enumerate(smart_chunks):
    print(f"\nChunk {i + 1} ({len(chunk)} chars):")
    print(chunk)