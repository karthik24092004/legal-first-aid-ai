from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
document_names = []

# Read all documents
for file in os.listdir("documents"):

    path = os.path.join("documents", file)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

        documents.append(text)
        document_names.append(file)

# Create embeddings
embeddings = model.encode(documents)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# Save index
faiss.write_index(
    index,
    "vectorstore/legal_index.faiss"
)

# Save document mapping
with open(
    "vectorstore/document_mapping.pkl",
    "wb"
) as f:
    pickle.dump(
        {
            "documents": documents,
            "names": document_names
        },
        f
    )

print("FAISS index created successfully!")