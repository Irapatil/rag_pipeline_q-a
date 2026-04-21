import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_PATH = "vector_store/index.faiss"
DOC_PATH = "vector_store/docs.pkl"


def load_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs


def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def run_ingestion():
    print("📥 Loading documents...")

    docs = load_documents("data/docs")

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))

    print(f"✂️ Created {len(all_chunks)} chunks")

    embeddings = model.encode(all_chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, VECTOR_PATH)

    with open(DOC_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print("✅ Vector store created successfully!")


if __name__ == "__main__":
    run_ingestion()