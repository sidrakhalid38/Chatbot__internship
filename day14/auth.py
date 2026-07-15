import secrets
import hashlib
import json
from pathlib import Path
from datetime import datetime

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


# API keys plain text mein store nahi hongi.
# Sirf unka SHA-256 hash JSON file mein save hoga.
KEYS_FILE = Path("day14/api_keys.json")


# Yeh endpoints API key ke baghair access ho sakte hain.
PUBLIC_PATHS = {
    "/health",
    "/docs",
    "/openapi.json",
    "/",
    "/redoc",
    "/admin/keys",
}


def _hash_key(raw_key: str) -> str:
    """
    Raw API key ka SHA-256 hash return karta hai.
    """
    return hashlib.sha256(raw_key.encode()).hexdigest()


def load_keys() -> dict:
    """
    JSON file se registered API keys load karta hai.
    File exist na kare to empty dictionary return hoti hai.
    """
    if not KEYS_FILE.exists():
        return {}

    try:
        return json.loads(KEYS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_keys(keys: dict) -> None:
    """
    API key registry ko JSON file mein save karta hai.
    """
    KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)

    KEYS_FILE.write_text(
        json.dumps(keys, indent=2),
        encoding="utf-8",
    )


def generate_api_key(label: str) -> str:
    """
    Secure random API key generate karta hai.

    Plain-text key sirf ek dafa return hoti hai.
    JSON file mein sirf iska SHA-256 hash save hota hai.
    """
    raw_key = "nxc-" + secrets.token_urlsafe(32)
    key_hash = _hash_key(raw_key)

    keys = load_keys()

    keys[key_hash] = {
        "label": label,
        "created_at": datetime.utcnow().isoformat(),
        "request_count": 0,
    }

    _save_keys(keys)

    return raw_key


def validate_key(raw_key: str) -> dict | None:
    """
    API key ko validate karta hai.

    Valid key par metadata return hota hai.
    Invalid key par None return hota hai.
    """
    key_hash = _hash_key(raw_key)
    keys = load_keys()

    if key_hash not in keys:
        return None

    keys[key_hash]["request_count"] += 1
    _save_keys(keys)

    return keys[key_hash]


def revoke_key(raw_key: str) -> bool:
    """
    API key registry se key remove karta hai.
    """
    key_hash = _hash_key(raw_key)
    keys = load_keys()

    if key_hash not in keys:
        return False

    del keys[key_hash]
    _save_keys(keys)

    return True


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Har protected request ka X-API-Key header check karta hai.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        raw_key = request.headers.get("X-API-Key")

        if not raw_key:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Missing X-API-Key header."
                },
            )

        metadata = validate_key(raw_key)

        if metadata is None:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Invalid or revoked API key."
                },
            )

        request.state.api_key_label = metadata["label"]

        return await call_next(request)


if __name__ == "__main__":
    # Previous test file remove karein.
    if KEYS_FILE.exists():
        KEYS_FILE.unlink()

    print("--- Generating two keys ---")

    key1 = generate_api_key("intern-test")
    key2 = generate_api_key("admin")

    print(f"Key 1 (intern-test): {key1}")
    print(f"Key 2 (admin): {key2}")

    print("\n--- Validating key 1 ---")

    meta = validate_key(key1)
    print(f"Result: {meta}")

    print("\n--- Validating a fake key ---")

    result = validate_key("nxc-this-is-not-a-real-key")
    print(f"Result: {result} (should be None)")

    print("\n--- Key registry on disk ---")

    print(json.dumps(load_keys(), indent=2))

    print("\n--- Revoking key 2 ---")

    print(f"Revoked: {revoke_key(key2)}")
    print(f"Keys remaining: {len(load_keys())} (should be 1)")

    print("\nAll auth tests passed.")