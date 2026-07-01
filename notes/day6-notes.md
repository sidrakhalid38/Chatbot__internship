# Day 6 Notes – Hybrid Search & Reciprocal Rank Fusion (RRF)

## Task 1 – BM25 vs Dense Retrieval

| Query | BM25 Top Source | Dense Top Source | Better Method |
|-------|------------------|------------------|---------------|
| What is NexusChat? | sample.pdf | sample.pdf | Dense |
| Professional plan 200 USD | sample2.txt | sample2.txt | BM25 |
| confidential data AI services policy | sample.docx | sample.docx | Hybrid |

---

## Observation Questions

### 1. Which method found the pricing document first?

BM25 found the pricing document first because it searches exact keywords.

---

### 2. Which method performed better for "What is NexusChat?"

Dense retrieval performed better because it understands semantic meaning instead of exact keywords.

---

### 3. What happens if the user asks "What does the product do?"

BM25 may miss the correct result because the exact keywords are different.
Dense retrieval can still find the correct answer using semantic similarity.

---

# Task 2

Hybrid Search combines BM25 and Dense Retrieval using Reciprocal Rank Fusion (RRF).

Advantages:

- Better ranking
- Better keyword matching
- Better semantic understanding
- More accurate answers

---

# Benchmark

| Metric | Dense | Hybrid |
|---------|-------|--------|
| Top Source | sample2.txt | sample2.txt |
| Finds sample2.txt | Yes | Yes |
| Answer Quality | 4/5 | 5/5 |

---

# Reflection Questions

## Did Hybrid outperform Dense?

Yes.
Hybrid combines keyword search with semantic search, making retrieval more accurate.

---

## What happens if k = 1?

Top-ranked results receive much higher importance than lower-ranked results.

---

## How can Hybrid Search scale to 100k documents?

- Indexed BM25
- Vector database
- Parallel retrieval
- Caching
- Approximate Nearest Neighbor (ANN)

---

## When should each retrieval method be used?

Dense Retrieval:
- Semantic questions
- Natural language queries

BM25:
- Product IDs
- Codes
- Exact keywords

Hybrid:
- Real-world chatbot applications
- Best overall retrieval quality