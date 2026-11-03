"""
Document Q&A (Retrieval-Augmented Generation).

PDFs are split into overlapping chunks, embedded with a sentence-transformer and
stored in ChromaDB. For each question, the closest chunks are retrieved and added
to the prompt so the model can answer from the document.
"""

import re
import uuid
from typing import List

import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from config import RAG_MAX_DISTANCE

# Small (~90 MB) model that maps text to 384-number vectors; loaded once at import
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready!")


# Persistent storage: documents survive server restarts (path is relative to the start folder)
print("Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
print("ChromaDB ready!")



def get_collection(client_id: str):
    """
    Get or create a ChromaDB collection for a client
    Each client has their own isolated collection
    Collection name = client API key identifier
    """
    # ChromaDB collection names must be alphanumeric + hyphens
    safe_id = re.sub(r'[^a-zA-Z0-9-]', '-', client_id)

    # Cosine distance: the standard measure for sentence embeddings (0 = same meaning).
    # Only applied when the collection is first created.
    return chroma_client.get_or_create_collection(
        name=safe_id,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for better results
    )

def clean_text(text: str) -> str:
    """Clean extracted PDF text"""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', ' ', text)

    # Remove lines that are mostly numbers/dots (table artifacts)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        non_numeric = re.sub(r'[\d\.\-\s\,]', '', line)

        # Under 15% real characters: treat as junk (page numbers, dotted lines)
        if len(line) > 0 and len(non_numeric) / len(line) < 0.15:
            continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks of chunk_size words."""
    text = clean_text(text)
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:  # skip tiny chunks
            chunks.append(chunk)

        # Overlap so a sentence cut at a boundary still appears whole in one chunk
        start += chunk_size - overlap

    return chunks


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF given as bytes; raises ValueError if unreadable."""
    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        raise ValueError(f"Could not read PDF: {str(e)}")
    return text.strip()

def store_document(client_id: str, conv_id: str, pdf_bytes: bytes, filename: str) -> int:
    """
    Process and store document in ChromaDB
    Organized by client_id — each client isolated
    Metadata tracks which conversation and file each chunk belongs to
    Returns the number of chunks stored.
    """
    # Extract and clean text
    text = extract_text_from_pdf(pdf_bytes)
    if not text:
        raise ValueError("No text found in PDF. It may be a scanned image.")

    # Split into chunks
    chunks = split_into_chunks(text)
    if not chunks:
        raise ValueError("Could not extract readable content from PDF.")

    # Get client collection
    collection = get_collection(client_id)

    # Embed all chunks at once (faster than one by one)
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

    # Generate unique IDs for each chunk
    chunk_ids = [str(uuid.uuid4()) for _ in chunks]

    # Store in ChromaDB with metadata
    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[
            {
                "filename": filename,
                "conv_id": conv_id,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
    )

    print(f"[RAG] Stored '{filename}' — {len(chunks)} chunks for client '{client_id}'")
    return len(chunks)

def retrieve_relevant_chunks(
    client_id: str,
    question: str,
    conv_id: str = None,
    top_k: int = 4
) -> str:
    """
    Search ChromaDB for most relevant chunks
    Optionally filter by conversation (specific uploaded docs)
    Returns a text block ready to add to the prompt, or "" if nothing relevant.
    """
    try:

        collection = get_collection(client_id)

        # Check if collection has any documents
        if collection.count() == 0:
            return ""

        # Embed the question with the same model used for the chunks
        question_embedding = embedder.encode([question]).tolist()

        # Build filter — optionally limit to specific conversation
        where_filter = {"conv_id": conv_id} if conv_id else None

        # Search ChromaDB
        results = collection.query(
            query_embeddings=question_embedding,
            n_results=min(top_k, collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Results are grouped per query; we sent one, so use [0]
        if not results["documents"] or not results["documents"][0]:
            return ""

        # Format results for injection into prompt
        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        formatted = "RELEVANT DOCUMENT CONTENT:\n---\n"

        for i, (chunk, meta, dist) in enumerate(zip(chunks, metadatas, distances), 1):
            # Only include reasonably relevant chunks (lower distance = more similar)
            print(f"[RAG] Chunk {i}, distance : {dist:.4f}, from file: {meta['filename']}")
            if dist < RAG_MAX_DISTANCE:
                formatted += f"[From: {meta['filename']} | Excerpt {i}]\n{chunk}\n---\n"

        if formatted == "RELEVANT DOCUMENT CONTENT:\n---\n":
            return ""
        
        formatted += "Use the document content above to answer question when it is relevant. If the question is not about the documents, answer normally from general knowledge."
        return formatted

    except Exception as e:
        print(f"[RAG] Retrieval error: {str(e)}")
        return ""

def has_documents(client_id: str, conv_id: str = None) -> bool:
    """Check if the client (optionally: this conversation) has uploaded documents."""
    try:
        collection = get_collection(client_id)
        if collection.count() == 0:
            return False
        if conv_id:
            results = collection.get(where={"conv_id": conv_id})
            return len(results["ids"]) > 0
        return True
    except:
        return False

def get_document_info(client_id: str, conv_id: str = None) -> dict:
    """Get list of uploaded documents for a client"""
    try:
        collection = get_collection(client_id)

        if collection.count() == 0:
            return {"files": [], "total_chunks": 0}

        # Get all metadata
        where_filter = {"conv_id": conv_id} if conv_id else None
        results = collection.get(
            where=where_filter,
            include=["metadatas"]
        )

        # Get unique filenames
        files = list(set(
            m["filename"] for m in results["metadatas"]
        ))

        return {
            "files": files,
            "total_chunks": len(results["ids"])
        }

    except Exception as e:
        return {"files": [], "total_chunks": 0, "error": str(e)}

def delete_document(client_id: str, filename: str, conv_id: str) -> dict:
    """Delete a specific document from ChromaDB"""
    try:
        collection = get_collection(client_id)
        # Only this file in this conversation
        collection.delete(
            where={
                "$and": [
                    {"filename": {"$eq": filename}},
                    {"conv_id": {"$eq": conv_id}}
                ]
            }
        )
        print(f"[RAG] Deleted '{filename}' for client '{client_id}'")
        return {"status": "deleted", "filename": filename}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def clear_all_documents(client_id: str) -> dict:
    """Delete all documents for a client"""
    try:
        safe_id = re.sub(r'[^a-zA-Z0-9-]', '-', client_id)
        chroma_client.delete_collection(safe_id)
        print(f"[RAG] Cleared all documents for client '{client_id}'")
        return {"status": "cleared"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
