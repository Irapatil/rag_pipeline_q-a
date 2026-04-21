import os
from openai import OpenAI

# Requires: pip install openai>=1.0.0
# Env: OPENAI_API_KEY, OPENAI_MODEL (optional, default gpt-4o-mini)

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def generate_answer(context: str, question: str) -> str:
    prompt = f"""You are a helpful assistant.
Answer ONLY using the provided context. If not found, say you don't know.

Context:
{context}

Question:
{question}

Answer:"""

    resp = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()