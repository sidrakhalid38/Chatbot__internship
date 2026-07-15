import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


# Maximum requests allowed in one time window
RATE_LIMIT_REQUESTS = 20

# Time window in seconds
RATE_LIMIT_WINDOW = 60


# Stores request timestamps separately for each client
_request_log: dict[str, deque] = defaultdict(deque)


# These paths will not be rate limited
EXEMPT_PATHS = {
    "/health",
    "/docs",
    "/openapi.json",
    "/",
    "/redoc",
    "/admin/keys",
}


def _get_client_id(request: Request) -> str:
    """
    API key label ko client ID ke taur par use karta hai.
    Agar label available na ho to IP address use hota hai.
    """
    label = getattr(
        request.state,
        "api_key_label",
        None,
    )

    if label:
        return label

    if request.client:
        return request.client.host

    return "unknown"


def is_rate_limited(
    client_id: str,
) -> tuple[bool, int]:
    """
    Sliding window rate limit check karta hai.

    Returns:
        (True, retry_after) agar limit exceed ho.
        (False, 0) agar request allowed ho.
    """
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW

    log = _request_log[client_id]

    # Purane timestamps remove karein
    while log and log[0] < cutoff:
        log.popleft()

    # Limit already reach ho chuki hai
    if len(log) >= RATE_LIMIT_REQUESTS:
        retry_after = int(
            log[0] + RATE_LIMIT_WINDOW - now
        ) + 1

        return True, max(retry_after, 1)

    # Request allow karke timestamp save karein
    log.append(now)

    return False, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Har client ko maximum 20 requests per 60 seconds allow karta hai.
    """

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        client_id = _get_client_id(request)

        limited, retry = is_rate_limited(
            client_id
        )

        if limited:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        "Rate limit exceeded — "
                        f"max {RATE_LIMIT_REQUESTS} "
                        f"requests per "
                        f"{RATE_LIMIT_WINDOW}s."
                    ),
                    "retry_after": retry,
                },
                headers={
                    "Retry-After": str(retry)
                },
            )

        return await call_next(request)


if __name__ == "__main__":
    CLIENT = "test-client"

    print(
        f"Sending 25 requests as client: "
        f"{CLIENT}"
    )

    print(
        f"Limit: {RATE_LIMIT_REQUESTS} "
        f"per {RATE_LIMIT_WINDOW}s"
    )

    print()

    allowed = 0
    blocked = 0

    for i in range(1, 26):
        limited, retry = is_rate_limited(
            CLIENT
        )

        if limited:
            blocked += 1

            print(
                f"Request {i:02d}: "
                f"BLOCKED "
                f"(retry in {retry}s)"
            )
        else:
            allowed += 1

            print(
                f"Request {i:02d}: ALLOWED"
            )

    print(
        f"\nSummary: "
        f"{allowed} allowed, "
        f"{blocked} blocked"
    )

    print(
        "Expected: 20 allowed, 5 blocked"
    )

    assert allowed == RATE_LIMIT_REQUESTS

    assert blocked == (
        25 - RATE_LIMIT_REQUESTS
    )

    print(
        "All rate limiter tests passed."
    )