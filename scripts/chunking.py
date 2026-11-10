def split_text(text, chunk_size, chunk_overlap):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be nonnegative and smaller than chunk_size."
        )

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append(chunk_text)

        if end == len(text):
            break

        start = end - chunk_overlap

    return chunks


def chunk_documents(documents, chunk_size, chunk_overlap):
    records = []

    for document in documents:
        page_chunks = split_text(
            document["text"],
            chunk_size,
            chunk_overlap,
        )

        for chunk_index, text in enumerate(page_chunks, start=1):
            records.append({
                "chunk_id": (
                    f"{document['source']}"
                    f":p{document['page_number']}"
                    f":c{chunk_index}"
                ),
                "text": text,
                "source": document["source"],
                "page_number": document["page_number"],
                "chunk_index": chunk_index,
            })

    return records