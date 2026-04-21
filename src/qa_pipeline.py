import os
from src.retriever import Retriever

# Choose provider via env: LLM_PROVIDER=openai | vertex | local
PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()

if PROVIDER == "openai":
    from src.llm_openai import generate_answer
elif PROVIDER == "vertex":
    from src.llm_vertex import generate_answer
else:
    # fallback local model (already in your project)
    from transformers import pipeline
    _local = pipeline("text-generation", model="distilgpt2")

    def generate_answer(context: str, question: str) -> str:
        prompt = f"""Answer based on context.\nContext:\n{context}\n\nQ: {question}\nA:"""
        out = _local(prompt, max_length=200, do_sample=True)[0]["generated_text"]
        return out

retriever = Retriever()

def answer_question(query: str):
    chunks = retriever.search(query, k=3)
    context = "\n".join(chunks)
    answer = generate_answer(context, query)
    return {"answer": answer, "context": chunks}