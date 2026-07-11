import json
import re
import time

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from day4.generator import (
    GENERATION_MODEL,
    client,
    extract_message_content,
)


# Filhal generation aur judging ke liye
# same free model use ho raha hai.
JUDGE_MODEL = GENERATION_MODEL


def clamp_score(value):
    """
    Score ko 0.0 aur 1.0 ke darmiyan rakhta hai.
    """

    try:
        score = float(value)
        return max(0.0, min(1.0, score))

    except (TypeError, ValueError):
        return 0.0


def zero_scores():
    return {
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "context_recall": 0.0,
    }


def extract_scores(raw):
    """
    Judge response se JSON scores extract karta hai.
    """

    if not raw:
        return None

    cleaned = raw.strip()

    # Markdown JSON fences remove karo
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    # Pehle direct JSON parse try karo
    try:
        data = json.loads(cleaned)

        return {
            "faithfulness": clamp_score(
                data.get("faithfulness")
            ),
            "answer_relevance": clamp_score(
                data.get("answer_relevance")
            ),
            "context_recall": clamp_score(
                data.get("context_recall")
            ),
        }

    except json.JSONDecodeError:
        pass

    # Response ke andar JSON object search karo
    json_match = re.search(
        r"\{[\s\S]*?\}",
        cleaned,
    )

    if json_match:
        try:
            data = json.loads(
                json_match.group()
            )

            return {
                "faithfulness": clamp_score(
                    data.get("faithfulness")
                ),
                "answer_relevance": clamp_score(
                    data.get("answer_relevance")
                ),
                "context_recall": clamp_score(
                    data.get("context_recall")
                ),
            }

        except json.JSONDecodeError:
            pass

    # Last fallback: regex se values nikalo
    faith_match = re.search(
        r'"?faithfulness"?\s*[:=]\s*'
        r'(\d+(?:\.\d+)?)',
        cleaned,
        re.IGNORECASE,
    )

    relevance_match = re.search(
        r'"?answer_relevance"?\s*[:=]\s*'
        r'(\d+(?:\.\d+)?)',
        cleaned,
        re.IGNORECASE,
    )

    recall_match = re.search(
        r'"?context_recall"?\s*[:=]\s*'
        r'(\d+(?:\.\d+)?)',
        cleaned,
        re.IGNORECASE,
    )

    if (
        faith_match
        and relevance_match
        and recall_match
    ):
        return {
            "faithfulness": clamp_score(
                faith_match.group(1)
            ),
            "answer_relevance": clamp_score(
                relevance_match.group(1)
            ),
            "context_recall": clamp_score(
                recall_match.group(1)
            ),
        }

    return None


def _call_judge(prompt, retries=2):
    """
    OpenRouter judge model ko call karta hai.
    Empty response ya temporary error par retry karta hai.
    """

    for attempt in range(1, retries + 1):
        try:
            print(
                "Judging with OpenRouter... "
                f"attempt {attempt}/{retries}"
            )

            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Evaluate the RAG output and return only compact JSON. "
                            "Do not explain. Do not show reasoning. "
                            "Use exactly these keys: "
                            "faithfulness, answer_relevance, context_recall."   
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=1000,
                timeout=90,
            )

            raw = extract_message_content(response)

            if not raw:
                print(
                    "Judge returned an empty response."
                )

                print(
                    "Response model:",
                    getattr(
                        response,
                        "model",
                        "Unknown",
                    ),
                )

                print(
                    "Finish reason:",
                    getattr(
                        response.choices[0],
                        "finish_reason",
                        "Unknown",
                    ),
                )

                if attempt < retries:
                    print(
                        "Waiting 7 seconds "
                        "before judge retry..."
                    )
                    time.sleep(7)
                    continue

                return zero_scores()

            print(
                "Judge raw response:",
                raw,
            )

            scores = extract_scores(raw)

            if scores is not None:
                return scores

            print(
                "Could not parse judge response."
            )

            if attempt < retries:
                print(
                    "Retrying judge after 7 seconds..."
                )
                time.sleep(7)
                continue

            return zero_scores()

        except RateLimitError as error:
            print(
                "OpenRouter judge rate limit reached."
            )
            print(f"Error: {error}")

            if attempt < retries:
                time.sleep(12)

        except APITimeoutError as error:
            print(
                "OpenRouter judge request timed out."
            )
            print(f"Error: {error}")

            if attempt < retries:
                time.sleep(7)

        except APIConnectionError as error:
            print(
                "Could not connect to OpenRouter "
                "during judging."
            )
            print(f"Error: {error}")

            if attempt < retries:
                time.sleep(7)

        except APIStatusError as error:
            print(
                "OpenRouter judge API error."
            )
            print(
                f"Status code: {error.status_code}"
            )
            print(f"Error: {error}")

            if attempt < retries:
                time.sleep(7)

        except Exception as error:
            print(
                "Unexpected judge error."
            )
            print(
                f"Error type: "
                f"{type(error).__name__}"
            )
            print(f"Error: {error}")
            print(
                f"Full error: {repr(error)}"
            )

            if attempt < retries:
                time.sleep(7)

    return zero_scores()


def evaluate_single(
    question,
    answer,
    context_chunks,
    ground_truth,
):
    context_text = "\n\n".join(
        (
            f"[Source: "
            f"{chunk.get('source', 'Unknown')}]\n"
            f"{chunk.get('text', '')}"
        )
        for chunk in context_chunks
    )

    prompt = f"""
Evaluate this RAG result.

Score every metric from 0.0 to 1.0.

METRICS:

Faithfulness:
Are the factual claims in the generated answer supported
by the retrieved context?

Answer relevance:
Does the generated answer directly answer the question?

Context recall:
Does the retrieved context contain the information required
to produce the ground-truth answer?

QUESTION:
{question}

GROUND-TRUTH ANSWER:
{ground_truth}

RETRIEVED CONTEXT:
{context_text}

GENERATED ANSWER:
{answer}

Return exactly this JSON structure:

{{
  "faithfulness": 0.0,
  "answer_relevance": 0.0,
  "context_recall": 0.0
}}
""".strip()

    scores = _call_judge(prompt)

    faithfulness = scores["faithfulness"]
    answer_relevance = scores[
        "answer_relevance"
    ]
    context_recall = scores["context_recall"]

    mean_score = round(
        (
            faithfulness
            + answer_relevance
            + context_recall
        )
        / 3,
        4,
    )

    return {
        "faithfulness": round(
            faithfulness,
            4,
        ),
        "answer_relevance": round(
            answer_relevance,
            4,
        ),
        "context_recall": round(
            context_recall,
            4,
        ),
        "mean": mean_score,
    }


def evaluate_dataset(
    chatbot_name,
    test_cases,
    retrieve_fn,
    generate_fn,
):
    print(f"\nEvaluating: {chatbot_name}")
    print(f"Test cases: {len(test_cases)}")
    print("-" * 50)

    all_scores = []

    for index, case in enumerate(
        test_cases,
        start=1,
    ):
        question = case["question"]
        ground_truth = case["ground_truth"]

        chunks = retrieve_fn(question)

        answer = generate_fn(
            question,
            chunks,
        )

        generation_failed = (
            "could not generate an answer"
            in answer.lower()
        )

        if generation_failed:
            print(
                "Generation failed, so judge "
                "call was skipped."
            )

            scores = {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_recall": 0.0,
                "mean": 0.0,
            }

        else:
            # Free model ko consecutive calls ke
            # darmiyan thora time dena zaroori hai.
            time.sleep(5)

            scores = evaluate_single(
                question=question,
                answer=answer,
                context_chunks=chunks,
                ground_truth=ground_truth,
            )

        all_scores.append(scores)

        display_question = question

        if len(display_question) > 50:
            display_question = (
                display_question[:50] + "..."
            )

        print(
            f"  [{index}/{len(test_cases)}] "
            f"Q: {display_question}"
        )

        print(
            f"    faithfulness="
            f"{scores['faithfulness']:.2f}  "
            f"relevance="
            f"{scores['answer_relevance']:.2f}  "
            f"recall="
            f"{scores['context_recall']:.2f}  "
            f"mean="
            f"{scores['mean']:.2f}"
        )

        # Next test case se pehle cooldown
        if index < len(test_cases):
            time.sleep(5)

    if not all_scores:
        return {
            "chatbot": chatbot_name,
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_recall": 0.0,
            "mean": 0.0,
        }

    avg = {
        "chatbot": chatbot_name,
        "faithfulness": round(
            sum(
                score["faithfulness"]
                for score in all_scores
            )
            / len(all_scores),
            4,
        ),
        "answer_relevance": round(
            sum(
                score["answer_relevance"]
                for score in all_scores
            )
            / len(all_scores),
            4,
        ),
        "context_recall": round(
            sum(
                score["context_recall"]
                for score in all_scores
            )
            / len(all_scores),
            4,
        ),
        "mean": round(
            sum(
                score["mean"]
                for score in all_scores
            )
            / len(all_scores),
            4,
        ),
    }

    print(
        "\n  AVERAGES -> "
        f"faithfulness="
        f"{avg['faithfulness']:.4f}  "
        f"relevance="
        f"{avg['answer_relevance']:.4f}  "
        f"recall="
        f"{avg['context_recall']:.4f}  "
        f"MEAN="
        f"{avg['mean']:.4f}"
    )

    return avg