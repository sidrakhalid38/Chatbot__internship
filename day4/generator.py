import os
import time

import certifi
import httpx
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY .env file mein nahi mili."
    )


# OpenRouter ke liye proper timeout configuration
timeout_config = httpx.Timeout(
    connect=20.0,
    read=90.0,
    write=30.0,
    pool=20.0,
)


# Shared OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=timeout_config,
    max_retries=0,
    http_client=httpx.Client(
        verify=certifi.where(),
        timeout=timeout_config,
    ),
)


GENERATION_MODEL = "openai/gpt-oss-20b:free"


def build_prompt(question, retrieved_chunks):
    context_lines = []

    for chunk in retrieved_chunks:
        source = chunk.get("source", "Unknown source")
        text = chunk.get("text", "")

        context_lines.append(
            f"[Source: {source}]\n{text}"
        )

    context_block = "\n\n".join(context_lines)

    prompt = f"""
You are a helpful RAG assistant.

Answer the question strictly using the provided context.

Rules:
1. Use only information from the CONTEXT.
2. Do not use outside knowledge.
3. If the answer is not present, say:
   "I could not find that information in the provided documents."
4. Keep the answer concise and direct.
5. End with a SOURCES section containing the filenames used.

CONTEXT:
{context_block}

QUESTION:
{question}

ANSWER:
""".strip()

    return prompt


def extract_message_content(response):
    """
    Safely extracts text from an OpenAI/OpenRouter response.
    """

    if response is None:
        return ""

    if not getattr(response, "choices", None):
        return ""

    message = response.choices[0].message

    if message is None:
        return ""

    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()

    # Kuch providers content ko list ke form mein return karte hain
    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")

                if text:
                    text_parts.append(str(text))

            else:
                text = getattr(item, "text", "")

                if text:
                    text_parts.append(str(text))

        return "\n".join(text_parts).strip()

    return ""


def generate_answer(
    question,
    retrieved_chunks,
    retries=2,
):
    prompt = build_prompt(
        question,
        retrieved_chunks,
    )

    for attempt in range(1, retries + 1):
        try:
            print(
                "Generating answer with OpenRouter... "
                f"attempt {attempt}/{retries}"
            )

            response = client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer only from the supplied context. "
                            "Be concise and factual."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=300,
                timeout=90,
            )

            answer = extract_message_content(response)

            if answer:
                return answer

            print("OpenRouter returned an empty answer.")

            print(
                "Response model:",
                getattr(response, "model", "Unknown"),
            )

            if attempt < retries:
                print(
                    "Waiting 5 seconds before retry..."
                )
                time.sleep(5)

        except RateLimitError as error:
            print("OpenRouter rate limit reached.")
            print(f"Error: {error}")

            if attempt < retries:
                time.sleep(10)

        except APITimeoutError as error:
            print("OpenRouter generation request timed out.")
            print(f"Error: {error}")

            if attempt < retries:
                time.sleep(5)

        except APIConnectionError as error:
            print(
                "Could not connect to OpenRouter."
            )
            print(f"Error: {error}")

            if attempt < retries:
                time.sleep(5)

        except APIStatusError as error:
            print(
                "OpenRouter returned an API error."
            )
            print(
                f"Status code: {error.status_code}"
            )
            print(f"Error: {error}")

            if attempt < retries:
                time.sleep(5)

        except Exception as error:
            print(
                "Unexpected generation error."
            )
            print(
                f"Error type: "
                f"{type(error).__name__}"
            )
            print(f"Error: {error}")

            if attempt < retries:
                time.sleep(5)

    return (
        "I could not generate an answer because "
        "OpenRouter returned an empty response "
        "or the API request failed."
    )


def display_answer(
    question,
    answer,
    retrieved_chunks,
):
    print("\n" + "=" * 65)
    print(f"QUESTION: {question}")
    print("=" * 65)

    print(f"\nANSWER:\n{answer}")

    print(
        "\n--- Retrieved chunks passed as context ---"
    )

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        source = chunk.get(
            "source",
            "Unknown source",
        )

        text = chunk.get("text", "")

        print(
            f"\nChunk {index} [{source}]:"
        )

        preview = text[:150]

        if len(text) > 150:
            print(f"{preview}...")
        else:
            print(preview)

    print("=" * 65)


if __name__ == "__main__":
    fake_chunks = [
        {
            "text": (
                "NexusChat supports PDF, "
                "DOCX, and TXT formats."
            ),
            "source": "sample.pdf",
        }
    ]

    test_question = (
        "What file formats does NexusChat support?"
    )

    test_answer = generate_answer(
        test_question,
        fake_chunks,
    )

    display_answer(
        test_question,
        test_answer,
        fake_chunks,
    )