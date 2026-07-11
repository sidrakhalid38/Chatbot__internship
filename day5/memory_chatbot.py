import os
import sys
import chromadb

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from day2.ingestion import load_document
from day2.chunking import recursive_chunking
from day3.embeddings import embed_text, embed_query
#from day4.generator import build_prompt
from day4.generator import generate_answer

load_dotenv()

CHUNK_SIZE = 300
CHUNK_OVERLAP = 40
TOP_K_RESULTS = 2
HISTORY_TURNS = 2
COLLECTION_NAME = "nexuschat_day5"

def rewrite_query(question, chat_history):
    if not chat_history:
        return question

    followup_words = ["it", "this", "that", "they", "its", "them"]
    question_words = question.lower().split()

    if any(word in question_words for word in followup_words):
        return chat_history[-1]["question"] + " " + question

    return question


def generate_with_memory(question, retrieved_chunks, chat_history):
    base_prompt = build_prompt(question, retrieved_chunks)

    if chat_history:
        history_lines = []
        for turn in chat_history[-HISTORY_TURNS:]:
            history_lines.append(f"User: {turn['question']}")
            history_lines.append(f"Assistant: {turn['answer']}")

        history_block = "\n".join(history_lines)

        base_prompt = base_prompt.replace(
            "Question:",
            f"Conversation so far:\n{history_block}\n\nQuestion:"
        )

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(base_prompt)

    return response.text


def safe_generate_with_memory(question, retrieved_chunks, chat_history):
    try:
        return generate_with_memory(question, retrieved_chunks, chat_history)
    except Exception as e:
        print(f"[Error calling Gemini: {e}]")
        return "Sorry, I ran into an error generating a response. Please try asking your question again."


def build_index(doc_paths, collection_name=COLLECTION_NAME):
    client = chromadb.Client()

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(collection_name)

    all_ids = []
    all_embeddings = []
    all_texts = []
    all_meta = []

    for path in doc_paths:
        text = load_document(path)
        chunks = recursive_chunking(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{os.path.basename(path)}_c{i}"
            embedding = embed_text(chunk)

            all_ids.append(chunk_id)
            all_embeddings.append(embedding)
            all_texts.append(chunk)
            all_meta.append({"source": os.path.basename(path)})

    collection.add(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_texts,
        metadatas=all_meta
    )

    print(f"Index complete — {len(all_ids)} chunks stored.")
    return collection


def retrieve(collection, query, top_k=TOP_K_RESULTS):
    query_vec = embed_query(query)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []

    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": text,
            "source": meta["source"],
            "distance": dist
        })

    return chunks


def run_chatbot_with_memory(doc_paths):
    collection = build_index(doc_paths)
    chat_history = []

    print("\n" + "=" * 65)
    print("NexusChat with memory is ready. Ask away.")
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

        standalone_question = rewrite_query(question, chat_history)

        if standalone_question != question:
            print(f'(rewritten for search as: "{standalone_question}")')

        chunks = retrieve(collection, standalone_question, top_k=TOP_K_RESULTS)

        if not chunks:
            answer = "I could not find any relevant information in the documents."
            print(f"\nNexusChat: {answer}")
            chat_history.append({"question": question, "answer": answer})
            continue

        answer = generate_answer(question, chunks)
        print(f"\nNexusChat: {answer}")

        chat_history.append({
            "question": question,
            "answer": answer
        })


if __name__ == "__main__":
    documents = [
        "day2/sample.txt",
        "day2/sample.pdf",
        "day2/sample.docx",
        "day2/sample2.txt",
    ]

    run_chatbot_with_memory(documents)