"""
Local AI Assistant: FastAPI server.

Runs a local LLM (Qwen3 via MLX) with streaming chat, per-conversation document
Q&A (RAG, see rag.py) and conversations saved to disk (see conversation_store.py).
"""

import json
import re
import uuid

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# MLX runs models on Apple Silicon (unified CPU/GPU memory)
from mlx_lm import generate, load, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from pydantic import BaseModel, Field

from config import API_KEYS, MODEL_NAME, SYSTEM_PROMPT_FILE
from dates import is_date_question, calculate_date_facts

from conversation_store import (
    delete_conversation,
    list_conversations,
    load_conversation,
    rename_conversation,
    save_conversation,
)

# --- Settings ---------------------------------------------------------------
# Importing rag loads the embedding model and opens ChromaDB
from rag import (
    clear_all_documents,
    delete_document,
    get_document_info,
    has_documents,
    retrieve_relevant_chunks,
    store_document,
)

# A conversation ID must be a lowercase UUID, e.g. 5c39c2ca-dd2d-4a6e-82a1-634dc0fe0af6
# ^ [0-9a-f]{8} - [0-9a-f]{4} - [0-9a-f]{4} - [0-9a-f]{4} - [0-9a-f]{12} $
UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

# No Hugging Face login needed: the library reads HF_TOKEN from the environment itself
app = FastAPI(title="My AI API")

# Allow browser requests from any origin (restrict this before deploying to a server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Model ------------------------------------------------------------------

def get_model_cache_path(model_name: str) -> Path:
    """Return the Hugging Face cache folder for a model, e.g. models--org--name."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    safe_name = model_name.replace("/", "--")
    return cache_dir / f"models--{safe_name}"




# Only informs the user; load() below downloads the model if needed
cache_path = get_model_cache_path(MODEL_NAME)
if cache_path.exists():
    print(f"Model found in cache — loading...")
else:
    print(f"Model not found in cache")
    print(f"Downloading {MODEL_NAME}...")
    print(f"This is a one-time download of approximately 5GB")
    print(f"Please wait...")


# Load once at startup, not per request
print("Loading model into memory...")
model, tokenizer = load(MODEL_NAME)

# Chat sampling: Qwen3's recommended settings for non-thinking mode
sampler = make_sampler(temp=0.7, top_p=0.8, top_k=20, min_p=0.0)

# Date extraction must be deterministic
sampler_zero = make_sampler(temp=0.0)

# Light repetition control
logits_processors = make_logits_processors(
    repetition_penalty=1.05,
    repetition_context_size=20
)

print("Model ready!")

# Active conversations in memory (conv_id -> messages); disk is the persistent copy
conversations = {}

class ChatRequest(BaseModel):
    message:str
    conversation_id:Optional[str] = Field(default=None, pattern=UUID_PATTERN, description="Optional conversation ID to continue an existing chat. If not provided, a new conversation is started.")

def validate_key(api_key:str):
    """
    Return the client name for a valid key, or reject the request with 401.
    """
    
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return API_KEYS[api_key]

def clean_response(response: str) -> str:
    """Remove Qwen3 <think> reasoning and cut off runaway number sequences."""
    response = re.sub(r'<think>[\s\S]*?</think>', '', response)
    response = re.sub(r'<think>[\s\S]*', '', response)  # unclosed think tag (reply cut off)

    # Remove runaway number sequences
    match = re.search(r'[\d\.\s]{100,}', response)
    if match:
        response = response[:match.start()].strip()

    return response.strip()

def system_prompt() -> str:
    """
    read system prompt file and replace {date} with today;s date 
	%A: Full weekday name (e.g., Tuesday).
	%d: Day of the month as a zero-padded decimal number (e.g., 03).
	%B: Full month name (e.g., November).
	%Y: Four-digit year (e.g., 2026).
    """
    text = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
    return text.replace("{today}",datetime.now().strftime("%A, %d %B %Y"))

def build_messages_with_context(
    conversations_list: list,
    user_question: str,
    conv_id: str = None,
    client_id: str = None) -> list:
    """
    System prompt + conversation, with document excerpts and date facts added to the latest question.
    """

    # make a copy 
    history = conversations_list.copy()
    
    #collect extra context and put it in correct order
    context = []
    if client_id and has_documents(client_id, conv_id):
        relevant = retrieve_relevant_chunks(client_id, user_question, conv_id)
        if relevant:
            print(f"[RAG] Injecting relevant chunks for: '{user_question}'")
            context.append(relevant)

    # Models are unreliable at calendar maths, so Python computes date facts
    if is_date_question(user_question):
        date_facts = calculate_date_facts(user_question,model_extractor=extract_date_with_model)
        context.append(f"ACCURATE DATE FACTS (use directly): {date_facts}")

    #put the context right before the question
    #a new dictionary so the stored messages are not modified
    if context and history and history[-1]["role"] == "user":
        question = history[-1]["content"]
        history[-1] = {"role":"user","content":"\n\n".join(context) + "\n\nQuestion:" + question}


    return  [{"role" : "system", "content":system_prompt()}] + history


# --- Date handling ----------------------------------------------------------

def extract_date_with_model(question: str):
    """Ask the model to find a date (fallback when regex finds none)."""

    print("[Date Extractor] Regex failed, trying model extraction...")

    extraction_messages = [
        {
            "role": "system",
            "content": (
                "You are a date extractor. "
                "Your ONLY job is to find a date in the user message. "
                "Reply with ONLY the date in DD/MM/YYYY format. "
                "If no specific date found reply with exactly: NONE "
                "Do not explain. Do not add any other text."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]
    extraction_prompt = tokenizer.apply_chat_template(
        extraction_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # Qwen3: answer directly, without hidden reasoning
    )

    # A date is only a few tokens; temperature 0 for consistent results
    raw = generate(
        model,
        tokenizer,
        prompt=extraction_prompt,
        max_tokens=15,
        sampler=sampler_zero
    )

    extracted = clean_response(raw)
    print(f"[Date Extractor] Model extracted: '{extracted}'")

    if not extracted or extracted.upper() == "NONE":
        return None
    try:
        return datetime.strptime(extracted, "%d/%m/%Y")
    except ValueError:
        return None


# --- Web page ---------------------------------------------------------------

app.mount("/scripts", StaticFiles(directory="scripts"), name="scripts")
app.mount("/styles", StaticFiles(directory="styles"), name="styles")

@app.get("/")
def root():
    return FileResponse("chat.html")


@app.get("/health")
def health(x_api_key: str = Header(...)):
    validate_key(x_api_key)
    return {"status": "healthy"}


# --- Chat -------------------------------------------------------------------



@app.post("/chat/stream")
def chat_stream(request:ChatRequest,x_api_key: str = Header(...)):
    """Stream a reply as Server-Sent Events: conv_id, then tokens, then done."""
    client_name = validate_key(x_api_key)

    # Continue an existing conversation or start a new one
    conv_id = request.conversation_id or str(uuid.uuid4())

    # Restore from disk if not in memory (e.g. after a server restart)
    if conv_id not in conversations:
        saved = load_conversation(x_api_key, conv_id)
        if saved:
            conversations[conv_id] = saved["messages"]
            print(f"[STORE] Restored conversation {conv_id} from disk")

    # New conversation starts empty system prompt will be inject while sending
    if conv_id not in conversations:
        conversations[conv_id] = []

    conversations[conv_id].append({
        "role":"user",
        "content": request.message
    })

    messages_to_send = build_messages_with_context(conversations[conv_id], request.message, conv_id, x_api_key)

    prompt = tokenizer.apply_chat_template(
        messages_to_send,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # Qwen3: answer directly, without hidden reasoning
    )

    def generate_stream():
        full_response = ""

        # Send the conversation ID first so a new chat learns it immediately
        yield f"data: {json.dumps({'type': 'conv_id', 'conversation_id': conv_id})}\n\n"

        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=800,
            sampler=sampler,
            logits_processors=logits_processors
        ):

            token = chunk.text
            full_response += token
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

        # Save the complete reply to memory and disk
        full_response = full_response.strip()
        conversations[conv_id].append({
            "role":"assistant",
            "content": full_response
        })
        save_conversation(x_api_key, conv_id, conversations[conv_id])

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    # X-Accel-Buffering: stops nginx (on a server) from buffering the stream
    return StreamingResponse(generate_stream(), media_type="text/event-stream",headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        })


# --- Documents --------------------------------------------------------------

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: str = Form(..., pattern=UUID_PATTERN),
    x_api_key: str = Header(...)
):
    """Upload a PDF and index it for the given conversation."""
    client_name = validate_key(x_api_key)

    # Only accept PDFs for now
    # if not file.filename.endswith(".pdf"):
    #     raise HTTPException(
    #         status_code=400,
    #         detail="Only PDF files supported currently"
    #     )

    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        chunk_count = store_document(x_api_key,conversation_id, pdf_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "uploaded",
        "filename": file.filename,
        "chunks": chunk_count,
        "conversation_id": conversation_id
    }

@app.get("/documents/{conv_id}")
def get_documents(conv_id: str= PathParam(pattern=UUID_PATTERN), x_api_key: str = Header(...)):
    """Get info about uploaded documents"""
    validate_key(x_api_key)
    return get_document_info(x_api_key, conv_id)


@app.delete("/documents/file/{filename}")
def remove_document(filename: str, conv_id: str = Header(... , pattern=UUID_PATTERN), x_api_key: str = Header(...)):
    validate_key(x_api_key)
    return delete_document(x_api_key, filename, conv_id)

@app.delete("/documents/all")
def remove_all_documents(x_api_key: str = Header(...)):
    validate_key(x_api_key)
    return clear_all_documents(x_api_key)


# --- Conversations ----------------------------------------------------------

@app.get("/conversations")
def get_conversations(x_api_key: str = Header(...)):
    """List all conversations for this client"""
    validate_key(x_api_key)
    return {"conversations": list_conversations(x_api_key)}

@app.get("/conversations/{conv_id}/load")
def load_conv(conv_id: str = PathParam(pattern=UUID_PATTERN), x_api_key: str = Header(...)):
    """Load a specific conversation"""
    validate_key(x_api_key)
    conv = load_conversation(x_api_key, conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Restore into memory
    conversations[conv_id] = conv["messages"]

    return {
        "conv_id": conv_id,
        "title": conv.get("title"),
        "messages": [
            m for m in conv["messages"]
            if m["role"] != "system"
        ]
    }

@app.delete("/conversations/{conv_id}")
def delete_conv(conv_id: str = PathParam(pattern=UUID_PATTERN), x_api_key: str = Header(...)):
    """Delete a conversation"""
    validate_key(x_api_key)
    if conv_id in conversations:
        del conversations[conv_id]
    delete_conversation(x_api_key, conv_id)
    return {"status": "deleted"}

@app.put("/conversations/{conv_id}/rename")
def rename_conv(
    body: dict,
    conv_id: str = PathParam(pattern=UUID_PATTERN),
    x_api_key: str = Header(...)
):
    """Rename a conversation"""
    validate_key(x_api_key)
    rename_conversation(x_api_key, conv_id, body.get("title", ""))
    return {"status": "renamed"}
