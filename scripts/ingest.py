"""Prepare one PDF for retrieval. Existing records require matching inputs."""

import argparse
import hashlib

from chromadb.errors import ChromaError

import config
from chunking import chunk_documents
from embeddings import embed_texts, load_embedding_model
from pdf_loader import load_pdf
from vector_store import open_collection, save_chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Replace the selected collection after preparing the PDF successfully.",
    )
    args = parser.parse_args()

    print("Reading PDF...")
    documents = load_pdf(config.PDF_PATH)
    metadata = {
        **config.pipeline_metadata(),
        "source_sha256": hashlib.sha256(config.PDF_PATH.read_bytes()).hexdigest(),
        "source_name": config.PDF_PATH.name,
    }
    chunks = chunk_documents(
        documents,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    if not chunks:
        raise ValueError("No usable chunks were extracted. Scanned PDFs may need OCR.")

    print(f"Page records: {len(documents)}")
    print(f"Chunk records: {len(chunks)}")
    model = load_embedding_model(config.EMBEDDING_MODEL_NAME)
    vectors = embed_texts(model, [chunk["text"] for chunk in chunks])

    collection = open_collection(
        config.VECTOR_DB_PATH, config.COLLECTION_NAME,
        metadata=metadata, rebuild=args.rebuild,
    )
    save_chunks(collection, chunks, vectors)
    print(f"Stored records: {collection.count()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, ChromaError) as error:
        print(f"Error: {error}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nIngestion interrupted. Run ingest.py again to complete it.")
        raise SystemExit(130)
