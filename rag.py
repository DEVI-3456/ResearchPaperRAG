import os
import fitz
import chromadb

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)


# --------------------------------------------------
# Models
# --------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_text(pdf_bytes):
    """
    Extract text from PDF page by page.
    Returns a list containing page number and text.
    """

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text("text")

        if text.strip():

            pages.append({
                "page": page_number + 1,
                "text": text.strip()
            })

    document.close()

    return pages


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def create_chunks(
    pages,
    chunk_size=500,
    overlap=100
):
    """
    Divide page text into overlapping chunks.

    chunk_size = number of words in each chunk
    overlap = number of overlapping words
    """

    chunks = []

    chunk_id = 0

    for page in pages:

        words = page["text"].split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_text = " ".join(
                words[start:end]
            )

            if chunk_text.strip():

                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "page": page["page"]
                })

                chunk_id += 1

            # Move forward while keeping overlap
            start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# CREATE VECTOR DATABASE
# --------------------------------------------------

def create_vector_database(chunks):
    """
    Create a ChromaDB collection and store
    embeddings of all chunks.
    """

    client = chromadb.Client()

    # Create a unique collection name
    collection = client.get_or_create_collection(
        name="research_paper"
    )

    # Remove previous data
    try:
        existing = collection.get()

        if existing["ids"]:

            collection.delete(
                ids=existing["ids"]
            )

    except Exception:
        pass

    documents = []
    embeddings = []
    ids = []
    metadatas = []

    for chunk in chunks:

        text = chunk["text"]

        # Generate embedding
        embedding = embedding_model.encode(
            text
        ).tolist()

        documents.append(text)

        embeddings.append(embedding)

        ids.append(
            str(chunk["id"])
        )

        metadatas.append({
            "page": chunk["page"]
        })

    # Add all chunks to vector database
    if documents:

        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

    return client, collection


# --------------------------------------------------
# SEMANTIC RETRIEVAL
# --------------------------------------------------

def retrieve_context(
    collection,
    question,
    top_k=5
):
    """
    Retrieve the most semantically relevant
    chunks for the user's question.
    """

    question_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(
            top_k,
            collection.count()
        )
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    retrieved_chunks = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        retrieved_chunks.append({
            "text": document,
            "page": metadata["page"]
        })

    return retrieved_chunks


# --------------------------------------------------
# GENERATE ANSWER USING GEMINI
# --------------------------------------------------

def generate_answer(
    question,
    retrieved_chunks
):
    """
    Generate a context-grounded answer
    using Gemini.
    """

    if not API_KEY:
        return (
            "Gemini API key is missing. "
            "Please add GEMINI_API_KEY to the .env file."
        )

    if not retrieved_chunks:
        return (
            "The answer is not available "
            "in the uploaded paper."
        )

    # Create context
    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"[Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a Research Paper Question Answering Assistant.

Your task is to answer the user's question using
ONLY the information provided in the context.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not supported by the context,
   say exactly:

   "The answer is not available in the uploaded paper."

4. Give a clear and concise answer.
5. When possible, mention the relevant page number.
6. Do not mention that you are an AI.
7. Do not make unsupported assumptions.

-------------------------
CONTEXT FROM RESEARCH PAPER
-------------------------

{context}

-------------------------
USER QUESTION
-------------------------

{question}

-------------------------
ANSWER
-------------------------
"""

    try:

        model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

        response = model.generate_content(
            prompt
        )

        return response.text

    except Exception as e:

        return (
            "Error generating answer: "
            + str(e)
        )


# --------------------------------------------------
# COMPLETE RAG PIPELINE
# --------------------------------------------------

def process_question(
    collection,
    question,
    top_k=5
):
    """
    Retrieve relevant chunks and generate
    a grounded answer.
    """

    retrieved_chunks = retrieve_context(
        collection,
        question,
        top_k
    )

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    pages = sorted(
        set(
            chunk["page"]
            for chunk in retrieved_chunks
        )
    )

    return answer, pages, retrieved_chunks