'''
Answer using retrieved chunks + Ollama 
'''

import sys
from openai import OpenAI

from config import LLM_MODEL_NAME, LLM_BASE_URL, LLM_API_KEY, TOP_K
from retrieve import retrieve_chunks


SYSTEM_PROMPT = """You are a research paper assistant.
Answer ONLY using the provided context from the paper.
If the context does not support an answer, say:
"Insufficient evidence in the indexed PDF."
Cite page numbers like (page 4). Do not invent facts."""


def format_context(hits):
    parts = []
    for i,h in enumerate(hits, 1):
        parts.append(
            f"[Chunk {i} | page {h['page']} \n {h['text']}\n]"
        )
    return "\n\n".join(parts)

def ask_paper(question):
    top_k = TOP_K
    hits = retrieve_chunks(question, top_k=TOP_K)
    if not hits:
        return "Insufficient evidence in the indexed PDF."
    prompt = f"""Context: {format_context(hits)} Question: {question} Answer in clear bullets. Cite page numbers."""
    client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY or "ollama"
    )

    resp = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the main contribution?"
    print(ask_paper(q))