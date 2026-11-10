from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name):
    return SentenceTransformer(
        model_name,
        device="cpu",
    )


def embed_texts(model, texts):
    if not texts:
        raise ValueError("No texts were provided.")

    token_counts = [
        len(model.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        ))
        for text in texts
    ]

    oversized_count = sum(
        count > model.max_seq_length
        for count in token_counts
    )

    if oversized_count:
        raise ValueError(
            f"{oversized_count} texts exceed the model input limit "
            f"of {model.max_seq_length} tokens."
        )

    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 1,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )