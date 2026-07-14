import json
from typing import Generator, List, Dict, Any

from day4.generator import stream_answer_tokens


def stream_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Generator[str, None, None]:
    """
    OpenRouter se streamed answer leta hai aur browser ke liye
    Server-Sent Events format mein events yield karta hai.

    Event types:

    data: {"token": "..."}
    data: {"sources": [...]}
    data: {"done": true}
    data: {"error": "..."}
    """

    try:
        # -------------------------------------------------
        # Stream answer tokens
        # -------------------------------------------------

        for token in stream_answer_tokens(
            question=question,
            retrieved_chunks=retrieved_chunks,
        ):
            event = {
                "token": token
            }

            yield (
                f"data: {json.dumps(event)}\n\n"
            )

        # -------------------------------------------------
        # Send sources after answer completes
        # -------------------------------------------------

        sources = []

        for chunk in retrieved_chunks:
            sources.append(
                {
                    "source": chunk.get(
                        "source",
                        "Unknown source",
                    ),
                    "preview": chunk.get(
                        "text",
                        "",
                    )[:150],
                }
            )

        yield (
            "data: "
            f"{json.dumps({'sources': sources})}"
            "\n\n"
        )

        # -------------------------------------------------
        # Final completion event
        # -------------------------------------------------

        yield (
            "data: "
            f"{json.dumps({'done': True})}"
            "\n\n"
        )

    except Exception as error:
        print(
            "Streaming generation error: "
            f"{type(error).__name__}: {error}"
        )

        yield (
            "data: "
            f"{json.dumps({'error': str(error)})}"
            "\n\n"
        )

        yield (
            "data: "
            f"{json.dumps({'done': True})}"
            "\n\n"
        )