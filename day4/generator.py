import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

GENERATION_MODEL = "models/gemini-2.5-flash"

def build_prompt(question, retrieved_chunks):
    context_lines = []

    for chunk in retrieved_chunks:
        source = chunk["source"]
        text = chunk["text"]
        context_lines.append(f"[Source: {source}]\n{text}")

    context_block = "\n\n".join(context_lines)

    prompt = f"""
You are a helpful assistant that answers questions strictly based on the context documents provided below.

Rules:
1. ONLY use information from the CONTEXT section.
2. Do not use your general knowledge.
3. If the answer is not in the context, say:
"I could not find that information in the provided documents."
4. Always end with a SOURCES section listing filenames used.
5. Be concise and direct.

--- CONTEXT ---
{context_block}
--- END CONTEXT ---

Question: {question}

Answer:
"""
    return prompt


def generate_answer(question, retrieved_chunks):
    prompt = build_prompt(question, retrieved_chunks)
    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt)
    return response.text


def display_answer(question, answer, retrieved_chunks):
    print("\n" + "=" * 65)
    print(f"QUESTION: {question}")
    print("=" * 65)
    print(f"\nANSWER:\n{answer}")

    print("\n--- Retrieved chunks passed as context ---")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"\nChunk {i} [{chunk['source']}]:")
        preview = chunk["text"][:150]
        print(f"{preview}..." if len(chunk["text"]) > 150 else chunk["text"])

    print("=" * 65)


if __name__ == "__main__":
    fake_chunks = [
        {
            "text": "NexusChat is an enterprise RAG chatbot that answers questions from your documents. It supports PDF, DOCX, and TXT formats.",
            "source": "sample.pdf",
        },
        {
            "text": "Users can ask natural language questions and receive cited answers. The system is built on Google Gemini and ChromaDB.",
            "source": "sample.pdf",
        },
        {
            "text": "All AI-generated content must be reviewed by a human before publication. Employees must not share confidential data with external AI services.",
            "source": "sample.docx",
        },
    ]

    q1 = "What file formats does NexusChat support?"
    answer1 = generate_answer(q1, fake_chunks)
    display_answer(q1, answer1, fake_chunks)

    q2 = "What is the pricing of NexusChat?"
    answer2 = generate_answer(q2, fake_chunks)
    display_answer(q2, answer2, fake_chunks)