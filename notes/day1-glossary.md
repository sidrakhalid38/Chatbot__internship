 Day 1 - RAG Glossary

 (Embedding)
Converts text into numerical vectors so AI can understand the meaning of text instead of just matching words.

 (Vector Store)
A database that stores embeddings and helps quickly find the most relevant information.

 (Chunking)
Breaking a large document into smaller pieces so the AI can retrieve only the relevant parts.

 (Context Window)
The maximum amount of text an AI model can process in a single prompt.

 (Hallucination)
When an AI generates information that is incorrect or made up.

---

 (Question 10)

Why can't we paste an entire document into the LLM prompt instead of chunking and retrieving?

Answer:

Large documents may exceed the model's context window and contain unnecessary information, making responses slower and less accurate.

---

 (Question 11)

What would go wrong if the retrieval step returns the wrong chunks?

Answer:

The AI may generate incorrect, misleading, or unrelated answers because it relies on the retrieved information.