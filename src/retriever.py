import faiss
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_PATH = "vector_store/index.faiss"
DOC_PATH = "vector_store/docs.pkl"


class Retriever:
    def __init__(self):
        self.index = faiss.read_index(VECTOR_PATH)

        with open(DOC_PATH, "rb") as f:
            self.docs = pickle.load(f)

    def search(self, query, k=3):
        query_vec = model.encode([query])

        distances, indices = self.index.search(query_vec, k)

        results = [self.docs[i] for i in indices[0]]

        return results