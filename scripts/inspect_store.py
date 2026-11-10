"""Inspect one stored record. The output may contain private document text."""

from chromadb.errors import ChromaError

import config
from vector_store import get_existing_collection


def main():
    collection = get_existing_collection(
        config.VECTOR_DB_PATH, config.COLLECTION_NAME,
        expected_metadata=config.pipeline_metadata(),
    )
    print("Stored records:", collection.count())
    records = collection.get(
        limit=1,
        include=["documents", "metadatas", "embeddings"],
    )
    if not records["ids"]:
        print("The collection is empty.")
        return

    print("ID:", records["ids"][0])
    print("Metadata:", records["metadatas"][0])
    print("Text preview:", records["documents"][0][:300])
    print("Vector dimensions:", len(records["embeddings"][0]))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, ChromaError) as error:
        print(f"Error: {error}")
        raise SystemExit(1)
