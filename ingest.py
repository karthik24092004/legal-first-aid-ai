from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
import re
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def create_chunks(text, chunk_size=8, overlap=2):
    sentences = split_into_sentences(text)

    chunks = []
    start = 0

    while start < len(sentences):
        end = start + chunk_size
        chunk = " ".join(sentences[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


all_chunks = []
chunk_sources = []

for file in os.listdir("documents"):
    if not file.endswith(".md"):
        continue

    path = os.path.join("documents", file)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = create_chunks(text)

    for chunk in chunks:
        all_chunks.append(chunk)
        chunk_sources.append(file)


embeddings = model.encode(all_chunks)
embeddings = np.array(embeddings).astype("float32")

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

os.makedirs("vectorstore", exist_ok=True)

faiss.write_index(
    index,
    "vectorstore/legal_index.faiss"
)

with open("vectorstore/chunks.pkl", "wb") as f:
    pickle.dump(
        {
            "chunks": all_chunks,
            "sources": chunk_sources
        },
        f
    )

print("Chunked FAISS index created successfully!")
print(f"Total chunks created: {len(all_chunks)}")