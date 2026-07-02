from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

print(f"Cross-encoder loaded: {CROSS_ENCODER_MODEL}")


def rerank(question, chunks, top_k=3):
    if not chunks:
        return []

    pairs = [[question, chunk["text"]] for chunk in chunks]
    scores = cross_encoder.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(
        chunks,
        key=lambda c: c["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]


def show_reranking(question, before_chunks, after_chunks):
    print(f"\nQuestion: {question}")

    print("\n[BEFORE re-ranking — retrieval order]")
    for i, c in enumerate(before_chunks, 1):
        score = c.get("rrf_score", c.get("distance", 0))
        print(f'{i}. [{c["source"]}] retrieval_score={score}')
        print(f'{c["text"][:100]}...')

    print("\n[AFTER re-ranking — cross-encoder order]")
    for i, c in enumerate(after_chunks, 1):
        print(f'{i}. [{c["source"]}] rerank_score={c["rerank_score"]:.4f}')
        print(f'{c["text"][:100]}...')


if __name__ == "__main__":
    fake_chunks = [
        {
            "text": "The weather in London is often rainy and overcast.",
            "source": "unrelated.txt",
            "rrf_score": 0.031,
        },
        {
            "text": "NexusChat supports PDF, DOCX, and TXT document formats for ingestion.",
            "source": "sample.pdf",
            "rrf_score": 0.028,
        },
        {
            "text": "Machine learning models learn patterns from large datasets.",
            "source": "sample.txt",
            "rrf_score": 0.025,
        },
        {
            "text": "NexusChat is an enterprise RAG chatbot that answers questions from your documents.",
            "source": "sample.pdf",
            "rrf_score": 0.022,
        },
        {
            "text": "Users can upload documents and query them using natural language.",
            "source": "sample.pdf",
            "rrf_score": 0.019,
        },
    ]

    question = "What is NexusChat and what can it do?"

    reranked = rerank(question, fake_chunks, top_k=3)

    show_reranking(question, fake_chunks, reranked)

    print("\n--- Key observation ---")
    print("Before re-ranking, the most relevant chunk was ranked 4th.")
    print("After re-ranking, it should appear in position 1 or 2.")
    print(
        f'Top chunk after re-ranking: [{reranked[0]["source"]}] '
        f'score={reranked[0]["rerank_score"]:.4f}'
    )