# Day 7 Notes — Re-Ranking & Query Decomposition

## Task 1 Observation Questions

### 1. Weather chunk vs NexusChat description score
Weather chunk score: very low and removed from top 3 after re-ranking.  
NexusChat description chunk score: 7.5527.

### 2. Best chunk position before and after re-ranking
Before re-ranking, the best NexusChat chunk was at position 4.  
After re-ranking, it moved to position 1.

### 3. Why is local CPU cross-encoder useful?
It avoids extra API calls, reduces cost, improves privacy, and can re-rank retrieved chunks locally.

---

## Task 3 Benchmark Table

| Metric | Week 1 Dense | Week 2 Hybrid | Week 2 Advanced |
|---|---|---|---|
| Mentions file formats? | Yes | Yes | Yes |
| Mentions pricing? | Partial | Yes | Yes |
| Gives recommendation? | No | Partial | Yes |
| Answer completeness (1–5) | 3 | 4 | 5 |

## Most complete version
Week 2 Advanced gave the most complete answer because it covered file formats, pricing plans, and recommendation for a 50-person team.

## Cost trade-off
Re-ranking adds local CPU inference, which may slow down each query slightly. Query decomposition adds multiple Gemini API calls, which increases cost and can hit rate limits. I would skip decomposition for simple questions and skip re-ranking when speed is more important than accuracy.

## How to catch bad decomposition
I would add validation checks to ensure each sub-question is self-contained, not repeated, and matches the original question. I could also show sub-questions to the user for confirmation in critical cases.

## One sentence for each technique

- Chunking solves the problem of large documents being too long to search or send to the model at once.
- Embedding solves the problem of converting text into numeric vectors for semantic search.
- Vector search solves the problem of finding chunks with similar meaning to the question.
- BM25 solves the problem of finding exact keyword matches.
- Hybrid search solves the problem of combining semantic search and keyword search.
- RRF solves the problem of merging results from multiple retrievers fairly.
- Re-ranking solves the problem of poor result ordering after retrieval.
- Query decomposition solves the problem of complex questions containing multiple information needs.
- Conversation memory solves the problem of understanding follow-up questions.
- Grounded generation solves the problem of producing answers based only on retrieved context.