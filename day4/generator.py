import os
import time

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

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

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


# ---------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------

def build_prompt(
    question,
    retrieved_chunks,
):
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

    context_block = "\n\n".join(
        context_parts
    )

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
# Response extraction
# ---------------------------------------------------------

def extract_message_content(response_data):
    """
    OpenRouter JSON response se safely answer extract karta hai.
    """

    if not isinstance(response_data, dict):
        return ""

    choices = response_data.get(
        "choices",
        [],
    )

    if not choices:
        return ""

    first_choice = choices[0]

    if not isinstance(first_choice, dict):
        return ""

    message = first_choice.get(
        "message",
        {},
    )

    if not isinstance(message, dict):
        return ""

    content = message.get(
        "content",
        "",
    )

    if isinstance(content, str):
        return content.strip()

    # Kuch providers content parts ki list return kar sakte hain.
    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get(
                    "text",
                    "",
                )

                if text:
                    text_parts.append(
                        str(text)
                    )

        return "\n".join(
            text_parts
        ).strip()

    return ""


# ---------------------------------------------------------
# OpenRouter request
# ---------------------------------------------------------

def generate_answer(
    question,
    retrieved_chunks,
    retries=2,
):
    """
    Requests library ke through OpenRouter se answer generate karta hai.
    """

    question = question.strip()

    if not question:
        return "Question cannot be empty."

    if not retrieved_chunks:
        return (
            "I could not find that information "
            "in the provided documents."
        )

    prompt = build_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    headers = {
        "Authorization": (
            f"Bearer {OPENROUTER_API_KEY}"
        ),
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "NexusChat Internship",
    }

    payload = {
        "model": GENERATION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied "
                    "document context. Be concise "
                    "and factual."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": 400,
    }

    last_error = ""

    for attempt in range(
        1,
        retries + 1,
    ):
        try:
            print(
                "Generating answer with OpenRouter... "
                f"attempt {attempt}/{retries}"
            )

            response = requests.post(
                OPENROUTER_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=(30, 120),
            )

            print(
                "OpenRouter status code:",
                response.status_code,
            )

            # -------------------------------------------------
            # Successful response
            # -------------------------------------------------

            if response.status_code == 200:
                try:
                    response_data = response.json()
                except ValueError:
                    last_error = (
                        "OpenRouter returned invalid JSON."
                    )

                    print(last_error)

                    if attempt < retries:
                        time.sleep(5)

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
                print(
                    "Raw response:",
                    response.text[:500],
                )

            # -------------------------------------------------
            # Rate limit
            # -------------------------------------------------

            elif response.status_code == 429:
                last_error = (
                    "OpenRouter free-model rate limit "
                    "or daily limit reached."
                )

                print(last_error)
                print(
                    "Response:",
                    response.text[:500],
                )

                # Retry se calls waste hongi.
                break

            # -------------------------------------------------
            # Authentication error
            # -------------------------------------------------

            elif response.status_code in {
                401,
                403,
            }:
                last_error = (
                    "OpenRouter API key was rejected."
                )

                print(last_error)
                print(
                    "Response:",
                    response.text[:500],
                )

                break

            # -------------------------------------------------
            # Other API errors
            # -------------------------------------------------

            else:
                last_error = (
                    "OpenRouter API returned status "
                    f"{response.status_code}."
                )

                print(last_error)
                print(
                    "Response:",
                    response.text[:500],
                )

                # Most 4xx errors retry se solve nahi hote.
                if 400 <= response.status_code < 500:
                    break

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
                f"Error type: "
                f"{type(error).__name__}"
            )
            print(f"Error: {error}")

        except Exception as error:
            last_error = (
                "Unexpected generation error."
            )

            print(last_error)
            print(
                f"Error type: "
                f"{type(error).__name__}"
            )
            print(f"Error: {error}")

        if attempt < retries:
            print(
                "Waiting 5 seconds before retry..."
            )

            time.sleep(5)

    print(
        "Answer generation failed. "
        f"Last error: {last_error}"
    )

    if "rate limit" in last_error.lower():
        return (
            "OpenRouter free-model rate limit or "
            "daily limit has been reached. "
            "Please try again later."
        )

    if "api key" in last_error.lower():
        return (
            "OpenRouter rejected the API key. "
            "Please check OPENROUTER_API_KEY "
            "in the .env file."
        )

    return (
        "I could not generate an answer because "
        "the OpenRouter request failed. "
        "Please try again."
    )


# ---------------------------------------------------------
# Console display helper
# ---------------------------------------------------------

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
# Direct test
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

    test_answer = generate_answer(
        question=test_question,
        retrieved_chunks=fake_chunks,
    )

    display_answer(
        question=test_question,
        answer=test_answer,
        retrieved_chunks=fake_chunks,
    )