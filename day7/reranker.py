import os

import certifi
from sentence_transformers import CrossEncoder


# Python, httpx aur Hugging Face ko valid CA certificate bundle use karwata hai.
os.environ.setdefault(
    "SSL_CERT_FILE",
    certifi.where(),
)

os.environ.setdefault(
    "REQUESTS_CA_BUNDLE",
    certifi.where(),
)

os.environ.setdefault(
    "CURL_CA_BUNDLE",
    certifi.where(),
)


CROSS_ENCODER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# Model import ke waqt load nahi hoga.
# Pehli rerank request par lazy load hoga.
cross_encoder = None
reranker_load_attempted = False


def get_cross_encoder():
    """
    Cross-encoder model ko sirf zarurat par load karta hai.

    Agar model SSL, internet ya download problem ki wajah se
    load na ho to None return karta hai. Is se API crash nahi hoti.
    """

    global cross_encoder
    global reranker_load_attempted

    if cross_encoder is not None:
        return cross_encoder

    # Har request par repeated download attempt se bachata hai.
    if reranker_load_attempted:
        return None

    reranker_load_attempted = True

    print("Loading cross-encoder reranker...")

    try:
        cross_encoder = CrossEncoder(
            CROSS_ENCODER_MODEL,
            max_length=512,
        )

        print(
            "Cross-encoder loaded: "
            f"{CROSS_ENCODER_MODEL}"
        )

        return cross_encoder

    except Exception as error:
        print(
            "Cross-encoder could not be loaded."
        )

        print(
            f"Error type: {type(error).__name__}"
        )

        print(f"Error: {error}")

        print(
            "Reranker fallback enabled. "
            "Original hybrid retrieval order will be used."
        )

        cross_encoder = None
        return None


def rerank(
    question,
    chunks,
    top_k=3,
):
    """
    Hybrid retriever ke candidate chunks ko CrossEncoder
    relevance score ke mutabiq dobara rank karta hai.

    Agar CrossEncoder load ya predict na kar sake to original
    hybrid retrieval order preserve hota hai.
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

    model = get_cross_encoder()

    # Model available na ho to safe fallback.
    if model is None:
        return valid_chunks[:top_k]

    pairs = [
        [
            question,
            chunk["text"],
        ]
        for chunk in valid_chunks
    ]

    try:
        scores = model.predict(
            pairs,
            show_progress_bar=False,
        )

    except Exception as error:
        print(
            "Cross-encoder reranking failed."
        )

        print(
            f"Error type: {type(error).__name__}"
        )

        print(f"Error: {error}")

        return valid_chunks[:top_k]

    scored_chunks = []

    for chunk, score in zip(
        valid_chunks,
        scores,
    ):
        updated_chunk = dict(chunk)

        updated_chunk["rerank_score"] = float(
            score
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
    Debugging ke liye retrieval aur reranked order show karta hai.
    """

    print(f"\nQuestion: {question}")

    print(
        "\n[BEFORE re-ranking - retrieval order]"
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
            f"retrieval_score={retrieval_score}"
        )

        if len(text) > 100:
            print(f"{preview}...")
        else:
            print(preview)

    print(
        "\n[AFTER re-ranking - final order]"
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
            "rerank_score"
        )

        text = chunk.get("text", "")
        preview = text[:100]

        if rerank_score is None:
            score_text = "fallback-order"
        else:
            score_text = (
                f"{rerank_score:.4f}"
            )

        print(
            f"{index}. [{source}] "
            f"rerank_score={score_text}"
        )

        if len(text) > 100:
            print(f"{preview}...")
        else:
            print(preview)


if __name__ == "__main__":
    fake_chunks = [
        {
            "text": (
                "The weather in London is often "
                "rainy and overcast."
            ),
            "source": "unrelated.txt",
            "rrf_score": 0.031,
        },
        {
            "text": (
                "NexusChat supports PDF, DOCX, "
                "and TXT document formats."
            ),
            "source": "sample.pdf",
            "rrf_score": 0.028,
        },
        {
            "text": (
                "NexusChat is an enterprise RAG "
                "chatbot that answers questions "
                "from documents."
            ),
            "source": "sample.pdf",
            "rrf_score": 0.022,
        },
    ]

    test_question = (
        "What is NexusChat and what can it do?"
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