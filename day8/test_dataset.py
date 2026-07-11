TEST_CASES = [
    {
        "question": "What is NexusChat?",
        "ground_truth": "NexusChat is an enterprise RAG chatbot that answers questions from documents. It supports PDF, DOCX, and TXT file formats and is built on Google Gemini and ChromaDB.",
    },
    {
        "question": "What file formats does NexusChat support?",
        "ground_truth": "NexusChat supports PDF, DOCX, and TXT file formats.",
    },
    {
        "question": "What is the company policy on sharing data with external AI services?",
        "ground_truth": "Employees must not share confidential data with external AI services.",
    },
    {
        "question": "What must happen before AI-generated content is published?",
        "ground_truth": "All AI-generated content must be reviewed by a human before publication.",
    },
    {
        "question": "How much does the Professional plan cost per month?",
        "ground_truth": "The Professional plan costs 200 USD per month.",
    },
    {
        "question": "How many users does the Starter plan support?",
        "ground_truth": "The Starter plan supports up to 10 users.",
    },
    {
        "question": "Is there a free trial available?",
        "ground_truth": "Yes, all plans include a 14-day free trial with no credit card required.",
    },
    {
        "question": "How does RAG allow AI to answer questions from specific documents?",
        "ground_truth": "RAG combines retrieval and generation: relevant document chunks are retrieved and passed as context to the language model, allowing it to answer questions grounded in specific documents rather than general training knowledge.",
    },
]