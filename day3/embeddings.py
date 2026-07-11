import math
import re
import hashlib

EMBED_DIM = 384


def _tokenize(text):
    text = text.lower()
    return re.findall(r"\b[a-z0-9]+\b", text)


def _hash_token(token):
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h, 16)


def embed_text(text):
    vector = [0.0] * EMBED_DIM
    tokens = _tokenize(text)

    if not tokens:
        return vector

    for token in tokens:
        index = _hash_token(token) % EMBED_DIM
        vector[index] += 1.0

    norm = math.sqrt(sum(v * v for v in vector))

    if norm == 0:
        return vector

    return [v / norm for v in vector]


def embed_query(text):
    return embed_text(text)


def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


if __name__ == "__main__":
    sentence = "Retrieval-Augmented Generation improves LLM accuracy."

    vector = embed_text(sentence)

    print(f"Text: {sentence}")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First 10 values: {[round(v, 5) for v in vector[:10]]}")
    print(f"Min value: {min(vector):.5f} Max value: {max(vector):.5f}")
    print()

    texts = [
        "The cat sat on the mat.",
        "A feline rested on a rug.",
        "Machine learning models process data.",
        "Deep neural networks learn representations.",
        "The weather in London is often rainy.",
    ]

    print('Cosine similarity to: "The cat sat on the mat."')
    print("*" * 50)

    base_vec = embed_text(texts[0])

    for text in texts:
        vec = embed_text(text)
        similarity = cosine_similarity(base_vec, vec)
        print(f"{similarity:.4f} | {text}")