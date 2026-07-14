# Day 13 Notes — Streaming Responses

## Latency Benchmark Results

Question used:

`What is NexusChat and what file formats does it support?`

Runs per method:

`1`

### Measured Results

- Non-streaming total response time: **16.34 seconds**
- Streaming time to first token (TTFT): **8.32 seconds**
- Streaming total response time: **9.37 seconds**
- Perceived speed improvement: **8.02 seconds**

Streaming improved the user experience because the first part of the answer appeared 8.02 seconds earlier than the complete non-streaming response.

---

## Reflection Question 1

### What was the measured TTFT compared with the non-streaming response time?

The measured non-streaming response time was 16.34 seconds. The streaming time to first token was 8.32 seconds. This means streaming showed the first content 8.02 seconds earlier than the non-streaming endpoint.

---

## Reflection Question 2

### Retrieval runs before streaming starts. How could retrieval feel faster?

The interface could immediately show a status message such as:

- Searching documents
- Retrieving relevant chunks
- Re-ranking results
- Generating answer

This would give the user visible progress while retrieval is running. Retrieval could also be optimized by reducing the number of candidate chunks, caching repeated queries, or using a faster re-ranking method.

---

## Reflection Question 3

### What happens if the user closes the browser during streaming?

The complete answer is saved to SQLite only after the stream finishes. If the user closes the browser tab during streaming, the connection may terminate before the done event is reached. In that case, the complete answer might not be saved to chat history.

For a production system, partial answers could be saved periodically or the generation process could run independently of the browser connection. The system could also save a status such as `incomplete` if streaming is interrupted.

---

## Reflection Question 4

### Should production keep both `/chat` and `/chat/stream`?

I would keep both endpoints.

The `/chat/stream` endpoint should be used by the normal web interface because it provides a better user experience.

The `/chat` endpoint is still useful for:

- automated tests
- integrations that do not support streaming
- simple API clients
- debugging
- fallback behavior

Keeping both endpoints provides compatibility for different clients while streaming remains the preferred user-facing option.

---

## Day 13 Summary

Day 13 added Server-Sent Events streaming to NexusChat.

The backend now provides:

- `POST /chat/stream`
- token-by-token OpenRouter streaming
- source events
- final done event with session ID
- SQLite history saving after completion

The frontend now provides:

- word-by-word answer rendering
- blinking streaming cursor
- source tags
- copy button
- session ID capture

The benchmark showed an 8.02-second perceived speed improvement.