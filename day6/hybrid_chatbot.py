import os
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from day5.memory_chatbot import rewrite_query
from day4.generator import generate_answer
from day6.hybrid_retriever import HybridRetriever

load_dotenv()


DOCUMENTS = [
    "day2/sample.txt",
    "day2/sample.pdf",
    "day2/sample.docx",
    "day2/sample2.txt",
]


def run_hybrid_chatbot():
    retriever = HybridRetriever()
    chat_history = []

    print("Building hybrid index (BM25 + dense)...")
    retriever.index(DOCUMENTS)

    print("\n" + "=" * 65)
    print("NexusChat Hybrid is ready.")
    print("Type 'quit' to exit.")
    print("=" * 65)

    while True:
        print()
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        standalone = rewrite_query(question, chat_history)

        if standalone != question:
            print(f'(search query: "{standalone}")')

        chunks = retriever.search(standalone, top_k=2, fetch_k=6)
        if not chunks:
            print("\nNexusChat: No relevant information found in documents.")
            chat_history.append({
                "question": question,
                "answer": "No relevant information found in documents."
            })
            continue

        print("Retrieved from:", [c["source"] for c in chunks])

        answer = generate_answer(question, chunks)    
        print(f"\nNexusChat: {answer}")

        chat_history.append({
            "question": question,
            "answer": answer
        })


if __name__ == "__main__":
    run_hybrid_chatbot()