import json
import time

import httpx


# ---------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------

BASE_URL = "http://localhost:8000"

QUESTION = (
    "What is NexusChat and what file formats does it support?"
)

# Daily OpenRouter limit 50 calls hai.
# 1 run means:
# 1 non-streaming call + 1 streaming call = 2 calls.
RUNS = 1


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def average(values):
    """
    Numeric values ka average return karta hai.
    """

    if not values:
        return 0.0

    return sum(values) / len(values)


# ---------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------

def run_benchmark():
    print("=" * 65)
    print("NexusChat Day 13 Latency Benchmark")
    print("=" * 65)

    print(f"\nQuestion: {QUESTION}")
    print(f"Runs per method: {RUNS}")

    print(
        "\nEstimated generation calls: "
        f"{RUNS * 2}"
    )

    # -----------------------------------------------------
    # Non-streaming benchmark
    # -----------------------------------------------------

    non_stream_times = []

    print("\n--- Non-streaming /chat ---")

    for run_number in range(1, RUNS + 1):
        start_time = time.perf_counter()

        response = httpx.post(
            f"{BASE_URL}/chat",
            json={
                "question": QUESTION
            },
            timeout=180,
        )

        elapsed = (
            time.perf_counter() - start_time
        )

        if response.status_code != 200:
            print(
                f"Run {run_number} failed: "
                f"HTTP {response.status_code}"
            )

            print(
                response.text[:500]
            )

            continue

        non_stream_times.append(
            elapsed
        )

        print(
            f"Run {run_number}: "
            f"{elapsed:.2f}s "
            "(full response received)"
        )

    # -----------------------------------------------------
    # Streaming benchmark
    # -----------------------------------------------------

    ttft_times = []
    total_stream_times = []

    print("\n--- Streaming /chat/stream ---")

    for run_number in range(1, RUNS + 1):
        start_time = time.perf_counter()
        first_token_time = None
        stream_error = None
        stream_completed = False

        try:
            with httpx.stream(
                method="POST",
                url=f"{BASE_URL}/chat/stream",
                json={
                    "question": QUESTION
                },
                timeout=180,
            ) as response:

                if response.status_code != 200:
                    print(
                        f"Run {run_number} failed: "
                        f"HTTP {response.status_code}"
                    )

                    response.read()

                    print(
                        response.text[:500]
                    )

                    continue

                for line in response.iter_lines():
                    if not line:
                        continue

                    if not line.startswith("data:"):
                        continue

                    json_text = (
                        line[len("data:"):]
                        .strip()
                    )

                    try:
                        event = json.loads(
                            json_text
                        )

                    except json.JSONDecodeError:
                        continue

                    if (
                        "token" in event
                        and first_token_time is None
                    ):
                        first_token_time = (
                            time.perf_counter()
                            - start_time
                        )

                    if event.get("error"):
                        stream_error = event.get(
                            "error"
                        )

                    if event.get("done"):
                        stream_completed = True
                        break

        except httpx.RequestError as error:
            print(
                f"Streaming run {run_number} "
                f"request failed: {error}"
            )

            continue

        total_time = (
            time.perf_counter() - start_time
        )

        if stream_error:
            print(
                f"Run {run_number} stream error: "
                f"{stream_error}"
            )

            continue

        if first_token_time is None:
            print(
                f"Run {run_number}: "
                "No token event was received."
            )

            continue

        ttft_times.append(
            first_token_time
        )

        total_stream_times.append(
            total_time
        )

        print(
            f"Run {run_number}: "
            f"TTFT={first_token_time:.2f}s | "
            f"total={total_time:.2f}s | "
            f"done={stream_completed}"
        )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    if not non_stream_times:
        print(
            "\nNon-streaming benchmark produced "
            "no successful result."
        )

        return

    if not ttft_times:
        print(
            "\nStreaming benchmark produced "
            "no successful result."
        )

        return

    non_stream_average = average(
        non_stream_times
    )

    ttft_average = average(
        ttft_times
    )

    total_stream_average = average(
        total_stream_times
    )

    perceived_improvement = (
        non_stream_average - ttft_average
    )

    print(
        "\nNon-streaming average "
        f"(user waits): {non_stream_average:.2f}s"
    )

    print(
        "Streaming TTFT average "
        f"(first content): {ttft_average:.2f}s"
    )

    print(
        "Streaming total average: "
        f"{total_stream_average:.2f}s"
    )

    print(
        "Perceived speed improvement: "
        f"{perceived_improvement:.2f}s"
    )

    if perceived_improvement > 0:
        print(
            "\nResult: Streaming showed the user "
            "content earlier than the normal endpoint."
        )

    else:
        print(
            "\nResult: No perceived improvement was "
            "measured in this run."
        )


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    run_benchmark()