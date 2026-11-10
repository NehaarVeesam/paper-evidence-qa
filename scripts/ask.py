from chromadb.errors import ChromaError

import config

from embeddings import embed_texts, load_embedding_model
from llm_client import generate_answer
from retrieval import build_context
from vector_store import get_existing_collection, search_chunks


def main():
    if not config.QWEN_BASE_URL or not config.QWEN_MODEL:
        raise ValueError("Set QWEN_BASE_URL and QWEN_MODEL in .env.")

    collection = get_existing_collection(
        config.VECTOR_DB_PATH,
        config.COLLECTION_NAME,
        expected_metadata=config.pipeline_metadata(),
    )

    if collection.count() == 0:
        raise ValueError("The collection is empty. Run ingest.py first.")

    print(f"Available passages: {collection.count()}")

    model = load_embedding_model(config.EMBEDDING_MODEL_NAME)

    print("Ready. Type /exit to quit.")

    while True:
        question = input("\nQuestion: ").strip()

        if question == "/exit":
            break

        if not question:
            continue

        try:
            question_vector = embed_texts(model, [question])[0]

            results = search_chunks(
                collection,
                question_vector,
                top_k=config.TOP_K,
            )

            print("\nRetrieved evidence:")

            for source_number, result in enumerate(results, start=1):
                chunk = result["chunk"]

                print(
                    f"[Source {source_number}] "
                    f"{chunk['source']}, "
                    f"PDF page {chunk['page_number']}, "
                    f"similarity {result['score']:.4f}"
                )

            context = build_context(results)

            response = generate_answer(
                question,
                context,
                base_url=config.QWEN_BASE_URL,
                api_key=config.QWEN_API_KEY,
                model_name=config.QWEN_MODEL,
                temperature=config.QWEN_TEMPERATURE,
                max_tokens=config.QWEN_MAX_TOKENS,
                timeout=config.QWEN_TIMEOUT,
                enable_thinking=config.QWEN_ENABLE_THINKING,
            )

            print("\nAnswer:")
            print(response["answer"] or "No final answer was returned.")
            print("\nUsage:", response["usage"])

            if response["finish_reason"] == "length":
                print("The answer may be incomplete: a token limit was hit.")

        except (ValueError, RuntimeError, ChromaError) as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, ChromaError) as error:
        print(f"Error: {error}")
        raise SystemExit(1)
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")
