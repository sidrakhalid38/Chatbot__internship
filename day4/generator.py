import json
import os
import time
from typing import Generator, List, Dict, Any

import requests
import truststore
from dotenv import load_dotenv


# ---------------------------------------------------------
# Windows SSL certificate support
# ---------------------------------------------------------

truststore.inject_into_ssl()


# ---------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY .env file mein nahi mili."
    )


# ---------------------------------------------------------
# OpenRouter configuration
# ---------------------------------------------------------

GENERATION_MODEL = "openai/gpt-oss-20b:free"

OPENROUTER_CHAT_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)

# Daily limit kam hai, is liye automatic retries default mein band hain.
DEFAULT_RETRIES = 1

REQUEST_CONNECT_TIMEOUT = 30
REQUEST_READ_TIMEOUT = 120

MAX_OUTPUT_TOKENS = 400


# ---------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------

def build_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    """
    Retrieved document chunks se grounded prompt banata hai.
    """

    context_parts = []

    for chunk in retrieved_chunks:
        source = chunk.get(
            "source",
            "Unknown source",
        )

        text = chunk.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        context_parts.append(
            f"[Source: {source}]\n{text}"
        )

    context_block = "\n\n".join(context_parts)

    return f"""
You are NexusChat, a helpful Retrieval-Augmented Generation assistant.

Answer the user's question strictly using the supplied document context.

Rules:
1. Use only information available in the CONTEXT.
2. Do not use outside knowledge.
3. Combine information from multiple chunks when needed.
4. If the answer is not present, say:
   "I could not find that information in the provided documents."
5. Keep the answer concise, clear, and factual.
6. Do not add a SOURCES section because the web interface displays sources separately.

CONTEXT:
{context_block}

QUESTION:
{question}

ANSWER:
""".strip()


# ---------------------------------------------------------
# Shared OpenRouter request helpers
# ---------------------------------------------------------

def build_headers() -> Dict[str, str]:
    """
    OpenRouter request headers return karta hai.
    """

    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "NexusChat Internship",
    }


def build_payload(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Streaming aur non-streaming requests ke liye shared payload.
    """

    prompt = build_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    return {
        "model": GENERATION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied document context. "
                    "Be concise and factual."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": stream,
    }


# ---------------------------------------------------------
# Non-streaming response extraction
# ---------------------------------------------------------

def extract_message_content(
    response_data: Dict[str, Any],
) -> str:
    """
    OpenRouter non-streaming JSON response se answer extract karta hai.
    """

    if not isinstance(response_data, dict):
        return ""

    choices = response_data.get("choices", [])

    if not choices:
        return ""

    first_choice = choices[0]

    if not isinstance(first_choice, dict):
        return ""

    message = first_choice.get("message", {})

    if not isinstance(message, dict):
        return ""

    content = message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    # Kuch providers content parts ki list return kar sakte hain.
    if isinstance(content, list):
        text_parts = []

        for item in content:
            if not isinstance(item, dict):
                continue

            text = item.get("text", "")

            if text:
                text_parts.append(str(text))

        return "\n".join(text_parts).strip()

    return ""


# ---------------------------------------------------------
# Streaming token extraction
# ---------------------------------------------------------

def extract_stream_token(
    event_data: Dict[str, Any],
) -> str:
    """
    OpenRouter streaming event ke delta object se token extract karta hai.
    """

    if not isinstance(event_data, dict):
        return ""

    choices = event_data.get("choices", [])

    if not choices:
        return ""

    first_choice = choices[0]

    if not isinstance(first_choice, dict):
        return ""

    delta = first_choice.get("delta", {})

    if not isinstance(delta, dict):
        return ""

    content = delta.get("content", "")

    if isinstance(content, str):
        return content

    # Kuch providers content ko parts ki list mein bhej sakte hain.
    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")

                if text:
                    text_parts.append(str(text))

            elif isinstance(item, str):
                text_parts.append(item)

        return "".join(text_parts)

    return ""


# ---------------------------------------------------------
# OpenRouter error helper
# ---------------------------------------------------------

def get_openrouter_error_message(
    status_code: int,
    response_text: str = "",
) -> str:
    """
    HTTP status code ko readable error message mein convert karta hai.
    """

    if status_code == 429:
        return (
            "OpenRouter free-model rate limit or daily limit "
            "has been reached. Please try again later."
        )

    if status_code in {401, 403}:
        return (
            "OpenRouter rejected the API key. "
            "Please check OPENROUTER_API_KEY in the .env file."
        )

    if status_code == 404:
        return (
            "The selected OpenRouter model or endpoint "
            "could not be found."
        )

    if status_code >= 500:
        return (
            "OpenRouter server is temporarily unavailable. "
            "Please try again later."
        )

    details = response_text[:300].strip()

    if details:
        return (
            f"OpenRouter API returned status {status_code}: "
            f"{details}"
        )

    return (
        f"OpenRouter API returned status {status_code}."
    )


# ---------------------------------------------------------
# Non-streaming answer generation
# ---------------------------------------------------------

def generate_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    retries: int = DEFAULT_RETRIES,
) -> str:
    """
    OpenRouter se complete non-streaming answer generate karta hai.

    Daily call limit bachane ke liye default retries = 1 hai.
    Ek normal question sirf ek generation API call use karta hai.
    """

    question = question.strip()

    if not question:
        return "Question cannot be empty."

    if not retrieved_chunks:
        return (
            "I could not find that information "
            "in the provided documents."
        )

    retries = max(1, retries)

    headers = build_headers()

    payload = build_payload(
        question=question,
        retrieved_chunks=retrieved_chunks,
        stream=False,
    )

    last_error = ""

    for attempt in range(1, retries + 1):
        try:
            print(
                "Generating answer with OpenRouter... "
                f"attempt {attempt}/{retries}"
            )

            response = requests.post(
                OPENROUTER_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=(
                    REQUEST_CONNECT_TIMEOUT,
                    REQUEST_READ_TIMEOUT,
                ),
            )

            print(
                "OpenRouter status code:",
                response.status_code,
            )

            if response.status_code != 200:
                last_error = get_openrouter_error_message(
                    status_code=response.status_code,
                    response_text=response.text,
                )

                print(last_error)

                # 4xx errors ko retry nahi karte, calls waste hongi.
                if 400 <= response.status_code < 500:
                    break

                if attempt < retries:
                    time.sleep(3)

                continue

            try:
                response_data = response.json()

            except ValueError:
                last_error = (
                    "OpenRouter returned invalid JSON."
                )

                print(last_error)

                if attempt < retries:
                    time.sleep(3)

                continue

            answer = extract_message_content(
                response_data
            )

            if answer:
                print(
                    "OpenRouter answer generated successfully."
                )

                return answer

            last_error = (
                "OpenRouter returned an empty answer."
            )

            print(last_error)

        except requests.exceptions.SSLError as error:
            last_error = (
                "OpenRouter SSL certificate error."
            )

            print(last_error)
            print(f"Error: {error}")

        except requests.exceptions.ConnectTimeout as error:
            last_error = (
                "Connection to OpenRouter timed out."
            )

            print(last_error)
            print(f"Error: {error}")

        except requests.exceptions.ReadTimeout as error:
            last_error = (
                "OpenRouter took too long to respond."
            )

            print(last_error)
            print(f"Error: {error}")

        except requests.exceptions.ConnectionError as error:
            last_error = (
                "Could not connect to OpenRouter."
            )

            print(last_error)
            print(f"Error: {error}")

        except requests.exceptions.RequestException as error:
            last_error = (
                "OpenRouter request failed."
            )

            print(last_error)
            print(
                f"Error type: {type(error).__name__}"
            )
            print(f"Error: {error}")

        except Exception as error:
            last_error = (
                "Unexpected generation error."
            )

            print(last_error)
            print(
                f"Error type: {type(error).__name__}"
            )
            print(f"Error: {error}")

        if attempt < retries:
            print(
                "Waiting 3 seconds before retry..."
            )

            time.sleep(3)

    print(
        "Answer generation failed. "
        f"Last error: {last_error}"
    )

    if last_error:
        return last_error

    return (
        "I could not generate an answer because "
        "the OpenRouter request failed. "
        "Please try again."
    )


# ---------------------------------------------------------
# Streaming answer generation
# ---------------------------------------------------------

def stream_answer_tokens(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Generator[str, None, None]:
    """
    OpenRouter se answer stream karta hai aur har received text
    chunk ko yield karta hai.

    Important:
    - Complete streaming response sirf ONE OpenRouter API call hai.
    - Har yielded token separate API call nahi hota.
    - Is function mein automatic retry nahi hai, daily limit bachane ke liye.
    """

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    if not retrieved_chunks:
        yield (
            "I could not find that information "
            "in the provided documents."
        )

        return

    headers = build_headers()

    payload = build_payload(
        question=question,
        retrieved_chunks=retrieved_chunks,
        stream=True,
    )

    print(
        "Starting OpenRouter streaming request..."
    )

    try:
        with requests.post(
            OPENROUTER_CHAT_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=(
                REQUEST_CONNECT_TIMEOUT,
                REQUEST_READ_TIMEOUT,
            ),
        ) as response:

            print(
                "OpenRouter streaming status code:",
                response.status_code,
            )

            if response.status_code != 200:
                error_message = get_openrouter_error_message(
                    status_code=response.status_code,
                    response_text=response.text,
                )

                raise RuntimeError(error_message)

            received_text = False

            # decode_unicode=True response bytes ko strings mein convert karta hai.
            for raw_line in response.iter_lines(
                decode_unicode=True
            ):
                if not raw_line:
                    continue

                line = raw_line.strip()

                # OpenRouter SSE format:
                # data: {"choices": [{"delta": {"content": "Hello"}}]}
                if not line.startswith("data:"):
                    continue

                data_text = line[len("data:"):].strip()

                # Streaming completion signal
                if data_text == "[DONE]":
                    break

                try:
                    event_data = json.loads(
                        data_text
                    )

                except json.JSONDecodeError:
                    print(
                        "Skipped invalid streaming JSON:",
                        data_text[:200],
                    )

                    continue

                # Provider stream ke andar error object bhej sakta hai.
                if event_data.get("error"):
                    error_data = event_data["error"]

                    if isinstance(error_data, dict):
                        error_message = error_data.get(
                            "message",
                            "OpenRouter streaming error.",
                        )
                    else:
                        error_message = str(error_data)

                    raise RuntimeError(error_message)

                token = extract_stream_token(
                    event_data
                )

                if token:
                    received_text = True
                    yield token

            if not received_text:
                raise RuntimeError(
                    "OpenRouter returned an empty streaming response."
                )

            print(
                "OpenRouter streaming completed successfully."
            )

    except requests.exceptions.SSLError as error:
        raise RuntimeError(
            "OpenRouter SSL certificate error."
        ) from error

    except requests.exceptions.ConnectTimeout as error:
        raise RuntimeError(
            "Connection to OpenRouter timed out."
        ) from error

    except requests.exceptions.ReadTimeout as error:
        raise RuntimeError(
            "OpenRouter took too long to stream the response."
        ) from error

    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            "Could not connect to OpenRouter."
        ) from error

    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            f"OpenRouter streaming request failed: {error}"
        ) from error


# ---------------------------------------------------------
# Console display helper
# ---------------------------------------------------------

def display_answer(
    question: str,
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> None:
    """
    Console mein answer aur retrieved chunks show karta hai.
    """

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

        text = chunk.get(
            "text",
            "",
        )

        print(
            f"\nChunk {index} [{source}]:"
        )

        preview = text[:150]

        if len(text) > 150:
            print(f"{preview}...")
        else:
            print(preview)

    print("=" * 65)


# ---------------------------------------------------------
# Direct non-streaming test
# ---------------------------------------------------------

if __name__ == "__main__":
    fake_chunks = [
        {
            "text": (
                "NexusChat uses ChromaDB as its persistent "
                "vector database for document embeddings. "
                "It uses SQLite to store persistent chat "
                "sessions and conversation history."
            ),
            "source": "nexuschat_database.txt",
        }
    ]

    test_question = (
        "What databases does NexusChat use?"
    )

    print("\nTesting non-streaming generation...\n")

    test_answer = generate_answer(
        question=test_question,
        retrieved_chunks=fake_chunks,
    )

    display_answer(
        question=test_question,
        answer=test_answer,
        retrieved_chunks=fake_chunks,
    )