'''
Retrieve top k chunks from ChromaDB for a user quesion
'''

from pathlib import Path
import sys

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    TOP_K
)

def get_collection():
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"ChromaDB directory not found at {CHROMA_DIR}")
    
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    return client.get_collection(name=CHROMA_COLLECTION_NAME, embedding_function=embedding_function)


def retrieve_chunks(question, top_k=5):
    '''
    Retrieve top k chunks from ChromaDB for a user quesion
    '''
    top_k = TOP_K if top_k is None else top_k
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")
    
    collection = get_collection()
    result = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["distances", "documents", "metadatas"]
    )

    documents = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]

    hits = []
    for doc, meta, dist in zip(documents, metas, distances):
        # cosine distance : 0 - identical, convert to a simple similarity score
        similarity = 1.0 - float(dist)
        hits.append({
            'text': doc,
            'page': meta['page'],
            'source': meta['source'],
            'score': similarity,
            'chunk_id': meta['chunk_id']
        })
    return hits

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the main contribution of the paper ?"
    hits = retrieve_chunks(q)
    print(f"Question: {q}\n")
    for i, h in enumerate(hits, start=1):
        print(f"--- Hit {i} | page {h['page']} | score {h['score']:.3f} ---")
        print(h["text"][:300].replace("\n", " "), "...\n")