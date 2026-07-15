# Day 14 — Authentication and Rate Limiting

## What I Implemented

Today I secured the NexusChat API using two protection layers:

1. API key authentication
2. Sliding-window rate limiting

The API now requires a valid `X-API-Key` header for protected endpoints. Public endpoints such as `/health`, `/docs`, and `/` remain accessible without a key.

Each API key is generated using a cryptographically secure random value. Only its SHA-256 hash is stored in `day14/api_keys.json`. The plain-text key is shown only once when it is generated.

The rate limiter allows a maximum of 20 requests per 60 seconds for each authenticated client. Requests exceeding the limit receive HTTP 429 with a `Retry-After` header and a `retry_after` value in the response body.

## Security Test Results

```text
Authentication:
- Missing API key returned HTTP 401
- Invalid API key returned HTTP 401
- Valid API key returned HTTP 200
- Health endpoint remained publicly accessible

Rate Limiting:
- First 20 requests returned HTTP 200
- Excess requests returned HTTP 429
- Retry-After response header was present
- retry_after field was present in the response body

Final Result:
11 passed, 0 failed