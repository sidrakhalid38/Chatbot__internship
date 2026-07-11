from langchain_text_splitters import RecursiveCharacterTextSplitter


#def fixed_size_chunking(text, size=200, overlap=50):
def recursive_chunking(text, size=300, overlap=40):
    chunks = []
    step = size - overlap

    for start in range(0, len(text), step):
        chunks.append(text[start:start + size])

    return chunks


def sentence_chunking(text, sentences_per_chunk=2):
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ". ".join(sentences[i:i + sentences_per_chunk]) + "."
        chunks.append(chunk)

    return chunks


def recursive_chunking(text, size=200, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)


def print_summary(name, chunks):
    avg_length = sum(len(chunk) for chunk in chunks) / len(chunks)

    print(f"\n{name}")
    print(f"Total Chunks: {len(chunks)}")
    print(f"Average Chunk Length: {avg_length:.2f}")

    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)


def compare_strategies(text, size=200, overlap=50, sentences_per_chunk=2):
    strategies = {
        "Fixed Size Chunking": fixed_size_chunking(text, size, overlap),
        "Sentence Based Chunking": sentence_chunking(text, sentences_per_chunk),
        "Recursive Chunking": recursive_chunking(text, size, overlap)
    }

    for name, chunks in strategies.items():
        print_summary(name, chunks)


if __name__ == "__main__":
    with open("day2/sample.txt", "r", encoding="utf-8") as file:
        text = file.read()

    print("Default Comparison")
    compare_strategies(text)

    print("\nExperiment 1: overlap=0")
    compare_strategies(text, overlap=0)

    print("\nExperiment 2: sentences=1")
    compare_strategies(text, sentences_per_chunk=1)

    print("\nExperiment 3: size=100")
    compare_strategies(text, size=100)