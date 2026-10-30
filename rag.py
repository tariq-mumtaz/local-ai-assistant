# Retrieval-Augmented Generation.
# The AI model doesn’t know what’s in PDFs, so before it answers, the code:
# Stores each PDF once: extracts the text, cuts it into pieces (“chunks”), 
# turns each chunk into numbers that represent its meaning (“embeddings”), 
# and saves them in ChromaDB.
# Retrieves for each question: turns the question into numbers the same way, 
# finds the chunks with the closest meaning, and adds them to the prompt 
# so the model can answer from them.

# fitz is PyMuPDF, the library that reads PDF files. 
# It’s installed as pymupdf but imported as fitz for historical reasons 
# (its engine was once called Fitz).
import fitz

# The ChromaDB library: the database that stores chunks and searches them by meaning.
import chromadb

# Regular expressions, used here to clean text and make safe names.
import re


# The library that turns text into embeddings. The next lines use it.
from sentence_transformers import SentenceTransformer


# Lets the code label return types, e.g. -> List[str] means “returns a list of strings”. 
# It’s only a label for readers and editors; it doesn’t change how the code runs.
from typing import List

# Creates a unique ID for every chunk stored in ChromaDB.
import uuid


# Loads the embedding model, all-MiniLM-L6-v2, and calls it embedder. 
# It’s a small model (about 90 MB), separate from Qwen3-8B, and it does one job: 
# turn text into a list of 384 numbers that captures its meaning. 
# Texts with similar meanings get similar numbers. 
# For example, “monthly rent” and “how much do I pay each month” end up close together even though they share no words.

#It’s loaded once, when api.py imports rag.py, and not on every request, because loading takes a few seconds. 
#That’s why these are the first lines you saw when the server started.

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready!")



# Initialize ChromaDB with persistent storage
# Data saved to disk — survives server restarts

# Opens the database and calls it chroma_client.
# PersistentClient means the data is saved to disk, so uploaded documents are still there after restart the server. 
# The alternative, chromadb.Client(), keeps everything in memory and loses it on restart.
# path="./chroma_db" means the data goes in a chroma_db folder inside the folder 
# we started the server from. ./ means “current folder”, 

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
    # ChromaDB only accepts certain characters in names, so this makes the key safe:
    # re.sub(pattern, replacement, text) finds everything matching the pattern and replaces it.
    # [^a-zA-Z0-9-] means “any character that is not a letter, digit or hyphen”. The ^ inside brackets means “not”.
    # Each such character becomes -.

    # So "dev-key-999" stays the same, while something like "client_key.001" would become "client-key-001".

    safe_id = re.sub(r'[^a-zA-Z0-9-]', '-', client_id)

    # get_or_create_collection opens the collection if it exists, or creates it 
    # if it doesn’t, so the code never has to check first.

    # metadata={"hnsw:space": "cosine"} sets how closeness is measured when searching:
    # HNSW is the search method ChromaDB uses. It’s a clever index that finds the nearest embeddings quickly without comparing against every single chunk.
    # cosine measures the angle between two embeddings, i.e. whether they point in the same direction of meaning, rather than raw distance. 
    # It’s the standard choice for sentence embeddings. Results range from 0 (same meaning) to 2 (opposite).

    return chroma_client.get_or_create_collection(
        name=safe_id,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for better results
    )

def clean_text(text: str) -> str:
    """Clean extracted PDF text"""
    # re is Python’s regular expressions module (imported at the top with import re).
    # .sub means substitute: find and replace.
    #  r before the quotes	“Raw” string: tells Python to pass backslashes \ through untouched, 
    # so the regex engine receives them as written. Regex patterns almost always start with r.
    # \n	A line break (pressing Enter)
    # {3,}	“The thing just before me, repeated 3 or more times”

    # So \n{3,} means “3 or more line breaks in a row”.
    # How { } works in general:

    # {3}: exactly 3
    # {3,}: 3 or more
    # {2,5}: between 2 and 5

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
        # Makes a copy of the line with digits (\d), dots, hyphens, whitespace (\s) and commas removed, 
        # so what’s left are the “real” characters such as letters.        
        non_numeric = re.sub(r'[\d\.\-\s\,]', '', line)

        # Compares how much is left: if under 15% of the line is real characters, 
        # it’s treated as junk and skipped.

        if len(line) > 0 and len(non_numeric) / len(line) < 0.15:
            continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    text = clean_text(text)

    # .split() with nothing in the brackets cuts the text at any whitespace (spaces, line breaks, tabs) 
    # and returns a list of words:
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        # end is where this chunk stops: 500 words after start
        end = start + chunk_size
        # words[start:end] is a slice: the words from position start up to (but not including) end. 
        # If end goes past the last word, Python simply stops at the end; no error.
        chunk = " ".join(words[start:end])
        # Keeps the chunk only if it’s longer than 50 characters (not words; len of a string counts characters). 
        # This drops tiny leftover scraps.
        if len(chunk.strip()) > 50:  # skip tiny chunks
            chunks.append(chunk)

        # Why overlap? Without it, a sentence like “The security deposit is AED 30,000” could be cut in half 
        # at a chunk boundary, with “The security deposit” in one chunk and “is AED 30,000” 
        # in the next, and neither chunk would contain the full answer. 
        # The 50-word overlap makes it likely that the sentence appears whole in at least one chunk.
        
        start += chunk_size - overlap

    return chunks


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract all text from a PDF
    Receives pdf_bytes: the raw file contents as received from the upload, not a file on disk. 
    bytes is Python’s type for raw binary data.
    If anything fails (e.g. the file isn’t really a PDF, or it’s damaged), 
    this catches the error and raises a new one: a ValueError with a readable message.
    raise deliberately creates an error.
    f"...{str(e)}" is an f-string: the f lets you insert values inside { }, 
    so the message includes the original error.
    """
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
    -> int means it returns a whole number: how many chunks were stored. 
    That’s the “chunks” number api.py sends back after an upload.
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
    # embedder.encode(chunks) is given the whole list and returns one row of 384 numbers per chunk. A 3-chunk document gives 3 rows × 384 numbers.
    # show_progress_bar=False stops it printing a progress bar in Terminal.
    # .tolist(): encode returns a NumPy array (a fast number table used by maths libraries). 
    # This converts it to a plain Python list of lists, which ChromaDB accepts.

    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

    # uuid.uuid4() creates a random unique ID like 3f2a9c1e-7b4d-....
    # str(...) turns it into text.
    # _ is a convention meaning “I need to loop this many times but don’t use the item itself”. 
    # It just makes one ID per chunk.

    # chunk_ids = []
    # for _ in chunks:
    #   chunk_ids.append(str(uuid.uuid4()))

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

    question: the user’s message.
    conv_id: str = None: the conversation. None means “nothing given”; then all of this client’s documents are searched.
    top_k: int = 4: return at most the 4 best matches.
    -> str: returns one block of text, ready to add to the prompt.
    """
    try:

        collection = get_collection(client_id)

        # Check if collection has any documents
        if collection.count() == 0:
            return ""

        # Embed the question
        # Turns the question into 384 numbers, with the same model used for the chunks, 
        # so the two can be compared. The question is put in a list ([question]) because encode and ChromaDB both work with lists, 
        # so the result is a list containing one row of numbers.
        question_embedding = embedder.encode([question]).tolist()

        # Build filter — optionally limit to specific conversation
        # This is a one-line if, written as value_if_true if condition else value_if_false
        where_filter = {"conv_id": conv_id} if conv_id else None

        # Search ChromaDB
        # query_embeddings: the question’s numbers.
        # n_results=min(top_k, collection.count()): how many results to return. min picks the smaller number, so if only 2 chunks exist it asks for 2, not 4.
        # where: the conversation filter.
        # include: what to return for each match: the text, the labels, and the distance 
        # (how far its meaning is from the question; smaller = closer).

        results = collection.query(
            query_embeddings=question_embedding,
            n_results=min(top_k, collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # ChromaDB can search for several questions at once, 
        # so it returns a list per question. With one question, the results are inside [0]. 
        # This line checks that something came back; if not, it returns empty text.
        
        if not results["documents"] or not results["documents"][0]:
            return ""

        # Format results for injection into prompt
        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        formatted = "RELEVANT DOCUMENT CONTENT:\n---\n"
        # chunks    = ["Monthly Rent: AED 15,000...", "Security Deposit: AED 30,000...", "Maintenance: Tenant..."]
        # metadatas = [{"filename": "lease.pdf"},    {"filename": "lease.pdf"},        {"filename": "lease.pdf"}]
        # distances = [0.21,                          0.48,                              0.73]

        # zip works like a zipper: it takes several lists and joins them position by position:

        # position 0 → ("Monthly Rent...",     {"filename": "lease.pdf"}, 0.21)
        # position 1 → ("Security Deposit...", {"filename": "lease.pdf"}, 0.48)
        # position 2 → ("Maintenance...",      {"filename": "lease.pdf"}, 0.73)

        # Each group in round brackets is a tuple: a small, fixed group of values, 
        # like a list that can’t be changed.

        # enumerate adds a counter to each item. The 1 means “start counting at 1” (by default it starts at 0):
        # The same thing written the long way

        # for position in range(len(chunks)):     # position = 0, 1, 2
        #     i     = position + 1                # counter starting at 1
        #     chunk = chunks[position]            # the text
        #     meta  = metadatas[position]         # the label
        #     dist  = distances[position]         # the distance
    
        for i, (chunk, meta, dist) in enumerate(zip(chunks, metadatas, distances), 1):
            # Only include reasonably relevant chunks (lower distance = more similar)
            if dist < 1.5:
                formatted += f"[From: {meta['filename']} | Excerpt {i}]\n{chunk}\n---\n"

        if formatted == "RELEVANT DOCUMENT CONTENT:\n---\n":
            return ""

        formatted += "Answer the user's question using ONLY the document content above. If the answer is not in the documents say so clearly."
        return formatted

    except Exception as e:
        print(f"[RAG] Retrieval error: {str(e)}")
        return ""

def has_documents(client_id: str, conv_id: str = None) -> bool:
    """
    Check if client has uploaded documents
    If a conversation is given, it fetches the chunks labelled with that conversation. 
    collection.get fetches stored items by filter, without a meaning search unlike query. 
    len(results["ids"]) > 0 is True if at least one chunk was found.
    """

    # ChromaDB always returns "ids" from collection.get, whatever we ask for. 
    # The include setting only controls the extra items (documents, metadatas, embeddings); 
    # the IDs always come back. 
    # If nothing matches the filter, you still get the key, just with an empty list:
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
    """
    Get list of uploaded documents for a client
    -> dict means it returns a dictionary.
    """
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
        # A 10-chunk file has 10 labels, all with the same filename. Read from the inside out:
        # m["filename"] for m in results["metadatas"] gets the filename from every label: lease.pdf, lease.pdf, lease.pdf, invoice.pdf...
        # set(...) is a collection that keeps only one of each: {lease.pdf, invoice.pdf}.
        # list(...) turns it back into a list, which can be sent to the browser as JSON.

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
        # Deletes every chunk matching both conditions. This is ChromaDB’s filter language:
        # "$eq" means “equals”: {"filename": {"$eq": "lease.pdf"}} means “filename equals lease.pdf”.
        # "$and" means “all of these must be true”, given as a list [ ] of conditions.

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