# Day 2 Notes: Document Ingestion & Chunking

## Experiment Results

| Strategy       | Setting             | Total Chunks | Average|  
                                                        Length 
|----------------|----------------------|-------------|-------:|
| Fixed Size     | overlap=50           | 3           |155.00  |
| Sentence Based | sentences=2          | 3           | 120.67 |
| Recursive      | size=200, overlap=50 | 3           | 120.33 |
| Fixed Size     | overlap=0            | 2           | ~232   |
| Sentence Based | sentences=1          | 5           | ~72    |
| Recursive      | size=100             | 5           | ~72    |

## Reflection Questions

### 1. Which chunking strategy worked best?
Recursive chunking worked best because it preserves meaningful text boundaries and avoids splitting important information in the middle of a sentence.

### 2. What happens when overlap is 0?
When overlap is 0, chunks do not share any repeated text. This reduces redundancy but may lose context between consecutive chunks.

### 3. What happens when sentences_per_chunk is 1?
Each sentence becomes its own chunk. This increases the number of chunks and keeps each chunk focused on a single idea.

### 4. What happens when chunk size is 100?
A smaller chunk size creates more chunks. This helps retrieve precise information but may split related content into multiple chunks.