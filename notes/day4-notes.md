# Day 4 Notes – Generation & Complete RAG Pipeline

## Observation Questions

### Q1. Did Gemini follow the "I could not find it" instruction?

Yes.

For the pricing question, Gemini responded:

> "I could not find that information in the provided documents."

It did not invent or hallucinate any pricing information.

---

### Q2. Did the SOURCES section correctly name sample.pdf?

Yes.

The answer correctly cited:

- sample.pdf

If the source label was not included in the prompt, Gemini would not know which document the information came from. The answer would lose traceability and users would not be able to verify the source.

---

### Q3. Why do we test generation with fake chunks before connecting ChromaDB?

Testing with fake chunks helps verify that:

- Prompt construction works correctly.
- Gemini generates grounded answers.
- Citation format is correct.
- Generation logic works independently from retrieval.

This catches prompt or generation bugs before debugging the vector database.

---

# Experiment A – Remove Grounding Instruction

Result:

After removing the grounding instruction, Gemini started using its own knowledge instead of relying only on the provided context.

This increased the chance of hallucinated answers.

Conclusion:

The grounding instruction is essential for a RAG system.

---

# Experiment B – Change top_k

top_k = 1

- Short answer
- Missed some details

top_k = 3

- Best balance
- Accurate
- Complete

top_k = 6

- More information
- Some repeated content
- Less concise

Best value:

**top_k = 3**

Reason:

It provides the most relevant information without unnecessary repetition.

---

# Experiment C – Add sample2.txt

Added:

day2/sample2.txt

Question:

What does the Professional plan cost?

Result:

The chatbot answered:

Professional plan costs **200 USD per month** and correctly cited:

sample2.txt

Conclusion:

Adding new knowledge to a RAG system is very easy.
Simply adding a new document and rebuilding the index makes the chatbot able to answer new questions.

---

# Reflection Questions

### Did removing the grounding instruction cause hallucination?

Yes.

Without grounding, Gemini may use its own knowledge instead of only the retrieved context.

---

### Which top_k gave the best results?

top_k = 3

It produced accurate, concise and relevant answers.

---

### Did the chatbot correctly cite sample2.txt?

Yes.

The pricing answer correctly cited sample2.txt.

---

### One thing this chatbot still cannot do

- Multi-turn conversation memory
- Persistent vector database
- Large-scale document indexing
- Web interface
- User authentication

---

### Which layer is hardest in a production RAG system?

Retrieval.

Reason:

If retrieval returns irrelevant chunks, the LLM cannot generate a correct answer even if the model itself is very powerful.

Good retrieval is the foundation of an accurate RAG system.

---
