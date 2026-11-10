import numpy as np


def retrieve_chunks(question_vector, passage_vectors, chunks, top_k=3):
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if len(passage_vectors) != len(chunks):
        raise ValueError("Each chunk must have one corresponding vector.")

    scores = passage_vectors @ question_vector

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "chunk": chunks[int(index)],
            "score": float(scores[index]),
        })

    return results


def build_context(results):
    parts = []

    for source_number, result in enumerate(results, start=1):
        chunk = result["chunk"]

        parts.append(
            f"[Source {source_number}]\n"
            f"Document: {chunk['source']}\n"
            f"PDF page: {chunk['page_number']}\n"
            f"Passage:\n{chunk['text']}"
        )

    return "\n\n".join(parts)