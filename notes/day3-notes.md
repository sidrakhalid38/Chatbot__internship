# Day 3 Notes - Embeddings

## Observation Questions

### 1. What is the dimension count of the Gemini embedding vector? What does each number represent?

The Gemini embedding model returned a vector of **3072 dimensions**. Each number represents one feature of the text's semantic meaning. Together, these values capture the overall meaning of the text.

---

### 2. Look at your cosine similarity scores. Did any result surprise you? Which two texts are more similar than you expected?

Yes. The sentences **"The cat sat on the mat."** and **"A feline rested on a rug."** had a cosine similarity of **0.8875**, even though they use different words. This shows that the embedding model understands semantic meaning rather than exact keywords.

---

### 3. Why do we use `task_type="RETRIEVAL_QUERY"` for the user's question instead of `RETRIEVAL_DOCUMENT`?

`RETRIEVAL_QUERY` is optimized for user search queries, while `RETRIEVAL_DOCUMENT` is optimized for stored documents. Using the correct task type improves retrieval accuracy.

---

### 4. If two chunks have a cosine similarity of 0.95, what does that tell you about their content?

A cosine similarity of **0.95** means the two chunks are extremely similar in meaning and discuss almost the same topic.

# Task 2 - Vector Store Observations

## 1. What happened to the search results after indexing complete documents?

After indexing the documents, ChromaDB was able to return the most relevant document chunks for each query instead of searching by keywords.

---

## 2. Why is embedding every chunk important?

Embedding every chunk converts text into vectors so that semantic search can find relevant information even if the exact words are different.

---

## 3. Why is metadata useful?

Metadata stores information such as the source filename. It helps identify which document the retrieved answer came from.

---

## 4. What is the difference between distance and similarity?

Similarity measures how close two vectors are. Distance measures how far apart they are. In ChromaDB, a lower distance means a more relevant result.

---

## 5. Which query produced the best results?

The query **"What are the rules about sharing data with AI tools?"** produced the best results because it correctly retrieved information from **sample.docx**, which contained the AI policy document.

---

## 6. Which query produced the weakest results?

The query **"What is the weather like in Paris?"** produced the weakest results because none of the indexed documents contained information about weather or Paris.

---

## 7. What did you learn from this task?

I learned how to create a ChromaDB collection, index document chunks with embeddings, perform semantic search, and retrieve the most relevant document chunks with their source and distance.

## task 3- pipeline diagram

RAW FILE (.txt / .pdf / .docx)
        |
        v
load_document()
        |
        v
recursive_chunking()
        |
        v
embed_text()
        |
        v
collection.add()  --> ChromaDB
        |
        | (Index complete)
        |
User Question
        |
        v
embed_query()
        |
        v
collection.query()
        |
        v
Most Relevant Chunks
        |
        v
(To be sent to Gemini on Day 4)
        |
        v
Final AI Answer