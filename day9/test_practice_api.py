import httpx

BASE = "http://localhost:8000"


def test_health():
    r = httpx.get(f"{BASE}/health")
    print(f"Health: {r.status_code} -> {r.json()}")


def test_echo():
    payload = {
        "message": "NexusChat",
        "repeat": 3
    }

    r = httpx.post(f"{BASE}/echo", json=payload)
    print(f"Echo: {r.status_code} -> {r.json()}")


def test_calculate():
    payload = {
        "a": 22,
        "b": 7,
        "operation": "divide"
    }

    r = httpx.post(f"{BASE}/calculate", json=payload)
    print(f"Calculate: {r.status_code} -> {r.json()}")


def test_bad_operation():
    payload = {
        "a": 1,
        "b": 2,
        "operation": "power"
    }

    r = httpx.post(f"{BASE}/calculate", json=payload)
    print(f"Bad operation: {r.status_code} -> {r.json()}")


def test_missing_field():
    payload = {
        "repeat": 2
    }

    r = httpx.post(f"{BASE}/echo", json=payload)
    print(f"Missing field: {r.status_code}")
    print(r.text)


if __name__ == "__main__":
    print("Running Practice API Tests...\n")

    test_health()
    test_echo()
    test_calculate()
    test_bad_operation()
    test_missing_field()

    print("\nAll tests complete.")