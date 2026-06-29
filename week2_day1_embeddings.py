from sentence_transformers import SentenceTransformer
import numpy as np

# Load a small, fast embedding model — runs locally, no API needed
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed some sentences
sentences = [
    "Patient has stage III ovarian cancer",
    "Tumor markers indicate malignant growth",
    "CA-125 levels elevated significantly",
    "Stock market hit record highs today",
    "The restaurant served excellent pasta",
    "Chemotherapy cycle completed successfully"
]

print("Generating embeddings...\n")
embeddings = model.encode(sentences)

print(f"Shape of one embedding: {embeddings[0].shape}")
print(f"Each sentence becomes {embeddings[0].shape[0]} numbers\n")

# Now find which sentences are most similar to a query
query = "cancer treatment progress"
query_embedding = model.encode([query])[0]

# Calculate similarity scores
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

print(f"Query: '{query}'\n")
print("Similarity scores:")

scores = []
for sentence, embedding in zip(sentences, embeddings):
    score = cosine_similarity(query_embedding, embedding)
    scores.append((score, sentence))

# Sort by most similar
scores.sort(reverse=True)
for score, sentence in scores:
    print(f"  {score:.3f} — {sentence}")