"""Persist chunks and search them using cosine distance."""

import chromadb
from chromadb.errors import NotFoundError


def validate_collection(collection, expected_metadata, require_complete=True):
    actual = collection.metadata or {}
    if any(actual.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError(
            "Stored data was built with a different document/configuration, "
            "or predates configuration checks. Run ingest.py --rebuild "
            "with your PDF available, or choose a new COLLECTION_NAME."
        )
    if require_complete and not actual.get("ingestion_complete", False):
        raise ValueError("Ingestion is incomplete. Run ingest.py again.")


def open_collection(db_path, collection_name, metadata, rebuild=False):
    client = chromadb.PersistentClient(path=str(db_path))
    if rebuild:
        try:
            client.delete_collection(name=collection_name)
        except NotFoundError:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
        metadata={**metadata, "ingestion_complete": False},
        configuration={"hnsw": {"space": "cosine"}},
    )
    validate_collection(collection, metadata, require_complete=False)
    return collection


def save_chunks(collection, chunks, vectors):
    if len(chunks) != len(vectors):
        raise ValueError("Each chunk must have one corresponding vector.")
    if not chunks:
        raise ValueError("No chunks were provided.")

    collection.modify(metadata={**collection.metadata, "ingestion_complete": False})
    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start:start + batch_size]
        batch_vectors = vectors[start:start + batch_size]

        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in batch_chunks],
            documents=[chunk["text"] for chunk in batch_chunks],
            metadatas=[
                {
                    "source": chunk["source"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                }
                for chunk in batch_chunks
            ],
            embeddings=batch_vectors.tolist(),
        )

    collection.modify(metadata={**collection.metadata, "ingestion_complete": True})


def search_chunks(collection, question_vector, top_k=3):
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    count = collection.count()
    if count == 0:
        raise ValueError("The collection is empty. Run ingest.py first.")

    response = collection.query(
        query_embeddings=[question_vector.tolist()],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for index, chunk_id in enumerate(response["ids"][0]):
        metadata = response["metadatas"][0][index]
        results.append({
            "chunk": {
                "chunk_id": chunk_id,
                "text": response["documents"][0][index],
                **metadata,
            },
            "score": 1.0 - response["distances"][0][index],
        })
    return results


def get_existing_collection(db_path, collection_name, expected_metadata=None):
    if not db_path.exists():
        raise ValueError("No vector database found. Run ingest.py first.")

    client = chromadb.PersistentClient(path=str(db_path))
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=None,
        )
    except NotFoundError:
        raise ValueError("Collection not found. Run ingest.py first.") from None

    if expected_metadata is not None:
        validate_collection(collection, expected_metadata)
    return collection
