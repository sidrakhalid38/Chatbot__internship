import httpx

BASE = "http://localhost:8000"

VALID_KEY = input("Paste your valid API key: ").strip()

passed = 0
failed = 0


def check(label, condition, detail=""):
    global passed, failed

    if condition:
        passed += 1
        print(f"PASS  {label}")
    else:
        failed += 1
        print(f"FAIL  {label} {detail}")


print("\n=== NexusChat Security Test Suite ===\n")

# ---------------- Authentication ----------------

print("[Authentication]")

r = httpx.post(
    f"{BASE}/chat",
    json={"question": "hi"},
)

check("No key -> 401", r.status_code == 401)

check(
    "No key message",
    "Missing X-API-Key" in r.json().get("detail", ""),
)

r = httpx.post(
    f"{BASE}/chat",
    json={"question": "hi"},
    headers={"X-API-Key": "nxc-totally-wrong-key"},
)

check("Wrong key -> 401", r.status_code == 401)

check(
    "Wrong key message",
    "Invalid" in r.json().get("detail", ""),
)

r = httpx.post(
    f"{BASE}/chat",
    json={"question": "What is NexusChat?"},
    headers={"X-API-Key": VALID_KEY},
    timeout=60,
)

check("Valid key -> 200", r.status_code == 200)

check(
    "Answer returned",
    "answer" in r.json(),
)

r = httpx.get(f"{BASE}/health")

check(
    "/health public",
    r.status_code == 200,
)

print()

# ---------------- Rate Limiting ----------------

print("[Rate Limiting]")
print("Sending 22 rapid requests to /sessions...")

statuses = []
responses = []

for _ in range(22):
    r = httpx.get(
        f"{BASE}/sessions",
        headers={"X-API-Key": VALID_KEY},
        timeout=10,
    )

    statuses.append(r.status_code)
    responses.append(r)

count_200 = statuses.count(200)
count_429 = statuses.count(429)

check(
    f"20 requests allowed (got {count_200})",
    count_200 == 20,
)

check(
    f"Excess requests blocked with 429 (got {count_429})",
    count_429 >= 2,
)

rate_limited_response = next(
    (
        response
        for response in responses
        if response.status_code == 429
    ),
    None,
)

if rate_limited_response is not None:
    check(
        "Retry-After header present",
        "retry-after" in rate_limited_response.headers,
    )

    check(
        "retry_after in body",
        "retry_after" in rate_limited_response.json(),
    )

print()
print(f"=== Results: {passed} passed, {failed} failed ===")
