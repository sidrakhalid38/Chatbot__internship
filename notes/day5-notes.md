# Day 5 Notes

## Observation 1
The chatbot successfully rewrote follow-up questions into standalone questions. For example, "What formats does it support?" was rewritten to "What formats does NexusChat support?", improving retrieval accuracy.

## Observation 2
If query rewriting is skipped, the chatbot may fail to retrieve the correct document because pronouns like "it" or "that" do not clearly identify the subject.

## Observation 3
The rewritten question is only used for document retrieval. The original user question is still passed to the generation step so the chatbot responds naturally.

## What I Learned
Today I learned how conversation memory improves a RAG chatbot. By combining chat history with query rewriting, the chatbot can understand follow-up questions and provide more accurate answers. I also learned how to make the chatbot more reliable using configuration variables and error handling.