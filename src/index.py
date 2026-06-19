'''
Embed chunks and persist them in ChromaDB

'''

from pathlib import Path
import sys

from load_pdf import load_pdf_pages
from chunk import chunk_pages

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    PDF_PATH
)


def build_chroma_index(pdf_path, reset=True):
    pdf_path = Path(pdf_path).resolve()
    pages = load_pdf_pages(pdf_path)
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)

    if not chunks:
        raise ValueError("No Chunks found")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception as e:
            print(f"Error deleting collection: {e}")
            print("Continuing with new collection")
    
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {"page": c["page"], "source": c["source"], "chunk_id": c["chunk_id"]}
        for c in chunks
    ]
    
    #Chroma has batch limits; add in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
    
    return len(chunks)


if __name__ == "__main__":
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PDF_PATH
    n = build_chroma_index(pdf_path)
    print(f"Indexed {n} chunks into Chroma Collection: {CHROMA_COLLECTION_NAME}")