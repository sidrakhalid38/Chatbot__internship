from sentence_transformers import CrossEncoder


CROSS_ENCODER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


print("Loading cross-encoder reranker...")

cross_encoder = CrossEncoder(
    CROSS_ENCODER_MODEL,
    max_length=512,
)

print(
    f"Cross-encoder loaded: "
    f"{CROSS_ENCODER_MODEL}"
)


def rerank(
    question,
    chunks,
    top_k=3,
):
    """
    Hybrid retriever ke candidate chunks ko
    CrossEncoder relevance score ke mutabiq
    dobara rank karta hai.

    Parameters:
        question:
            User ka query/question.

        chunks:
            Retrieved chunk dictionaries ki list.
            Har chunk mein kam az kam "text"
            aur "source" keys honi chahiye.

        top_k:
            Final kitne best chunks return karne hain.

    Returns:
        Re-ranked chunks ki list.
    """

    if not chunks:
        return []

    if top_k <= 0:
        return []

    valid_chunks = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()

        if not text:
            continue

        valid_chunks.append(chunk)

    if not valid_chunks:
        return []

    pairs = [
        [
            question,
            chunk["text"],
        ]
        for chunk in valid_chunks
    ]

    try:
        scores = cross_encoder.predict(
            pairs,
            show_progress_bar=False,
        )

    except Exception as error:
        print(
            "Cross-encoder reranking failed."
        )
        print(
            f"Error type: "
            f"{type(error).__name__}"
        )
        print(f"Error: {error}")

        # Fallback:
        # Agar reranker fail ho jaye to original
        # retrieval order preserve kar do.
        return valid_chunks[:top_k]

    scored_chunks = []

    for chunk, score in zip(
        valid_chunks,
        scores,
    ):
        updated_chunk = dict(chunk)

        updated_chunk["rerank_score"] = (
            float(score)
        )

        scored_chunks.append(
            updated_chunk
        )

    reranked_chunks = sorted(
        scored_chunks,
        key=lambda chunk: chunk[
            "rerank_score"
        ],
        reverse=True,
    )

    return reranked_chunks[:top_k]


def show_reranking(
    question,
    before_chunks,
    after_chunks,
):
    """
    Debugging ke liye retrieval order aur
    reranked order display karta hai.
    """

    print(f"\nQuestion: {question}")

    print(
        "\n[BEFORE re-ranking — "
        "retrieval order]"
    )

    for index, chunk in enumerate(
        before_chunks,
        start=1,
    ):
        source = chunk.get(
            "source",
            "Unknown source",
        )

        retrieval_score = chunk.get(
            "rrf_score",
            chunk.get(
                "distance",
                0.0,
            ),
        )

        text = chunk.get("text", "")
        preview = text[:100]

        print(
            f"{index}. [{source}] "
            f"retrieval_score="
            f"{retrieval_score}"
        )

        if len(text) > 100:
            print(f"{preview}...")
        else:
            print(preview)

    print(
        "\n[AFTER re-ranking — "
        "cross-encoder order]"
    )

    for index, chunk in enumerate(
        after_chunks,
        start=1,
    ):
        source = chunk.get(
            "source",
            "Unknown source",
        )

        rerank_score = chunk.get(
            "rerank_score",
            0.0,
        )

        text = chunk.get("text", "")
        preview = text[:100]

        print(
            f"{index}. [{source}] "
            f"rerank_score="
            f"{rerank_score:.4f}"
        )

        if len(text) > 100:
            print(f"{preview}...")
        else:
            print(preview)


if __name__ == "__main__":
    fake_chunks = [
        {
            "text": (
                "The weather in London is "
                "often rainy and overcast."
            ),
            "source": "unrelated.txt",
            "rrf_score": 0.031,
        },
        {
            "text": (
                "NexusChat supports PDF, "
                "DOCX, and TXT document "
                "formats for ingestion."
            ),
            "source": "sample.pdf",
            "rrf_score": 0.028,
        },
        {
            "text": (
                "Machine learning models "
                "learn patterns from large "
                "datasets."
            ),
            "source": "sample.txt",
            "rrf_score": 0.025,
        },
        {
            "text": (
                "NexusChat is an enterprise "
                "RAG chatbot that answers "
                "questions from documents."
            ),
            "source": "sample.pdf",
            "rrf_score": 0.022,
        },
        {
            "text": (
                "Users can upload documents "
                "and query them using natural "
                "language."
            ),
            "source": "sample.pdf",
            "rrf_score": 0.019,
        },
    ]

    test_question = (
        "What is NexusChat and "
        "what can it do?"
    )

    reranked_chunks = rerank(
        question=test_question,
        chunks=fake_chunks,
        top_k=3,
    )

    show_reranking(
        question=test_question,
        before_chunks=fake_chunks,
        after_chunks=reranked_chunks,
    )

    print("\n--- Key observation ---")

    if reranked_chunks:
        top_chunk = reranked_chunks[0]

        print(
            "Top chunk after re-ranking:"
        )

        print(
            f'[{top_chunk.get("source", "Unknown")}] '
            f'score='
            f'{top_chunk.get("rerank_score", 0.0):.4f}'
        )

    else:
        print(
            "No chunks were returned "
            "after reranking."
        )