import os
import google.generativeai as genai

# Requires: pip install google-generativeai
# Env: GOOGLE_API_KEY, GOOGLE_MODEL (optional, default gemini-1.5-flash)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
_MODEL = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")

_model = genai.GenerativeModel(_MODEL)

def generate_answer(context: str, question: str) -> str:
    prompt = f"""Answer the question ONLY from the context.
If not found, say you don't know.

Context:
{context}

Question:
{question}

Answer:"""

    resp = _model.generate_content(prompt)
    return (resp.text or "").strip()