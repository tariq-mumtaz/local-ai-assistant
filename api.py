"""
Local AI Assistant: FastAPI server.

Runs a local LLM (Qwen3 via MLX) with streaming chat, per-conversation document
Q&A (RAG, see rag.py) and conversations saved to disk (see conversation_store.py).
"""

import json
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# MLX runs models on Apple Silicon (unified CPU/GPU memory)
from mlx_lm import generate, load, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from pydantic import BaseModel

from config import API_KEYS, MODEL_NAME
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
    conversation_id:Optional[str] = None

def validate_key(api_key:str):
    """Return the client name for a valid key, or reject the request with 401."""
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

def build_messages_with_context(
    conversations_list: list,
    user_question: str,
    conv_id: str = None,
    client_id: str = None) -> list:
    """Add relevant document excerpts and date facts to the messages for this reply."""

    # Copy so injected context is used for this reply only, not saved to history
    messages = conversations_list.copy()

    # RAG: inject relevant document chunks if documents uploaded
    if client_id and has_documents(client_id, conv_id):
        relevant = retrieve_relevant_chunks(client_id, user_question, conv_id)
        if relevant:
            print(f"[RAG] Injecting relevant chunks for: '{user_question}'")
            messages.insert(1, {
                "role": "system",
                "content": relevant
            })

    # Models are unreliable at calendar maths, so Python computes date facts
    if is_date_question(user_question):
        date_facts = calculate_date_facts(user_question)
        messages.insert(1, {
            "role": "system",
            "content": f"ACCURATE DATE FACTS (use directly): {date_facts}"
        })

    return messages


# --- Date handling ----------------------------------------------------------

ARABIC_MONTHS = {
    # Gulf Arabic
    "يناير": 1, "فبراير": 2, "مارس": 3,
    "أبريل": 4, "مايو": 5, "يونيو": 6,
    "يوليو": 7, "أغسطس": 8, "سبتمبر": 9,
    "أكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
    # Levantine Arabic
    "كانون الثاني": 1, "شباط": 2, "آذار": 3,
    "نيسان": 4, "أيار": 5, "حزيران": 6,
    "تموز": 7, "آب": 8, "أيلول": 9,
    "تشرين الأول": 10, "تشرين الثاني": 11, "كانون الأول": 12
}

ARABIC_DAYS = {
    "Monday": "الاثنين",
    "Tuesday": "الثلاثاء",
    "Wednesday": "الأربعاء",
    "Thursday": "الخميس",
    "Friday": "الجمعة",
    "Saturday": "السبت",
    "Sunday": "الأحد"
}


def detect_language(text: str) -> str:
    """Return "arabic" if more than 30% of characters are Arabic, else "english"."""
    arabic_chars = sum(1 for c in text if '؀' <= c <= 'ۿ')
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0:
        return "english"
    return "arabic" if (arabic_chars / total_chars) > 0.3 else "english"

def convert_arabic_numerals(text: str) -> str:
    """Convert Arabic-Indic numerals ٠١٢٣ to Western 0123"""
    arabic_indic = "٠١٢٣٤٥٦٧٨٩"
    western = "0123456789"
    for a, w in zip(arabic_indic, western):
        text = text.replace(a, w)
    return text

def is_date_question(question: str) -> bool:
    """Detect questions about dates (keywords in English/Arabic, or a year 20xx)."""
    keywords = [
        # English
        "what day", "which day", "what date",
        "day is", "day will", "day was",
        "when is", "when will", "when was",
        "how many days", "days until", "days between",
        "days from", "days ago", "next week",
        "next month", "next year", "tomorrow",
        "yesterday", "day after", "day before",
        "falls on", "what is the day",
        # Arabic
        "ما هو اليوم", "أي يوم", "كم يوم",
        "متى", "غداً", "غدا", "أمس",
        "الأسبوع القادم", "الشهر القادم", "السنة القادمة"
    ]

    has_year = bool(re.search(r'\b20\d{2}\b', question))
    has_date_word = any(kw in question.lower() for kw in keywords)

    return has_date_word or has_year


def extract_date_with_regex(question: str):
    """Fast regex extraction before trying model"""

    # Normalize Arabic numerals first
    question = convert_arabic_numerals(question)

    months_en = {
        "january": 1, "february": 2, "march": 3,
        "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9,
        "october": 10, "november": 11, "december": 12
    }

    # "30th April 2026" or "30 April 2026"
    pattern = r'(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})'
    match = re.search(pattern, question, re.IGNORECASE)

    if match:
        try:
            day = int(match.group(1))
            month = months_en[match.group(2).lower()]
            year = int(match.group(3))
            return datetime(year, month, day)
        except ValueError:
            pass  # invalid date, e.g. 31 February

    # "April 30 2026" or "April 30th 2026"
    pattern2 = r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\d{4})'
    match2 = re.search(pattern2, question, re.IGNORECASE)
    if match2:
        try:
            month = months_en[match2.group(1).lower()]
            day = int(match2.group(2))
            year = int(match2.group(3))
            return datetime(year, month, day)
        except ValueError:
            pass

    # dd/mm/yyyy or dd-mm-yyyy (day first)
    pattern3 = r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})'
    match3 = re.search(pattern3, question)
    if match3:
        try:
            return datetime(int(match3.group(3)), int(match3.group(2)), int(match3.group(1)))
        except ValueError:
            pass

    return None


def extract_date_with_model(question: str):
    """
    Try regex first (fast + reliable for standard formats)
    Fall back to model for complex/Arabic phrasing
    """

    regex_result = extract_date_with_regex(question)
    if regex_result:
        print(f"[Date Extractor] Regex found: {regex_result.strftime('%d/%m/%Y')}")
        return regex_result

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
        add_generation_prompt=True
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

def calculate_date_facts(question: str) -> str:
    """
    Python calculates all date facts — never wrong
    Returns a string of facts to inject into prompt
    """
    today = datetime.now()
    lang = detect_language(question)
    result = []

    # Today as reference
    if lang == "arabic":
        result.append(f"اليوم هو {ARABIC_DAYS[today.strftime('%A')]}، {today.strftime('%d %B %Y')}.")
    else:
        result.append(f"Today is {today.strftime('%A, %d %B %Y')}.")

    # Extract specific date from question
    target_date = extract_date_with_model(question)

    if target_date:
        day_name = target_date.strftime("%A")
        diff = (target_date - today).days
        formatted = target_date.strftime("%d %B %Y")

        if lang == "arabic":
            arabic_day = ARABIC_DAYS.get(day_name, day_name)
            result.append(f"{formatted} يقع في يوم {arabic_day}.")
            if diff > 0:
                result.append(f"بعد {diff} يوم من اليوم.")
            elif diff < 0:
                result.append(f"مضت {abs(diff)} يوم.")
            else:
                result.append("هذا هو اليوم.")
        else:
            result.append(f"{formatted} falls on a {day_name}.")
            if diff > 0:
                result.append(f"That is {diff} days from today.")
            elif diff < 0:
                result.append(f"That was {abs(diff)} days ago.")
            else:
                result.append("That is today.")

    # Handle relative date keywords
    q = question.lower()

    if "tomorrow" in q or "غداً" in question or "غدا" in question:
        t = today + timedelta(days=1)
        if lang == "arabic":
            result.append(f"غداً هو {ARABIC_DAYS[t.strftime('%A')]}، {t.strftime('%d %B %Y')}.")
        else:
            result.append(f"Tomorrow is {t.strftime('%A, %d %B %Y')}.")

    if "yesterday" in q or "أمس" in question:
        y = today - timedelta(days=1)
        if lang == "arabic":
            result.append(f"أمس كان {ARABIC_DAYS[y.strftime('%A')]}، {y.strftime('%d %B %Y')}.")
        else:
            result.append(f"Yesterday was {y.strftime('%A, %d %B %Y')}.")

    if "next week" in q or "الأسبوع القادم" in question:
        nw = today + timedelta(weeks=1)
        if lang == "arabic":
            result.append(f"هذا اليوم الأسبوع القادم: {ARABIC_DAYS[nw.strftime('%A')]}، {nw.strftime('%d %B %Y')}.")
        else:
            result.append(f"This day next week is {nw.strftime('%A, %d %B %Y')}.")

    if "next year" in q or "السنة القادمة" in question:
        ny = today.replace(year=today.year + 1)
        if lang == "arabic":
            result.append(f"هذا اليوم السنة القادمة: {ARABIC_DAYS[ny.strftime('%A')]}، {ny.strftime('%d %B %Y')}.")
        else:
            result.append(f"This day next year is {ny.strftime('%A, %d %B %Y')}.")

    return " ".join(result)

def build_messages_with_date(conversations_list: list, user_question: str) -> list:
    """
    NEW: Intercept date questions and inject Python-calculated facts
    Returns messages list ready to send to model
    """
    messages = conversations_list.copy()

    if is_date_question(user_question):
        print(f"[Date Interceptor] Date question detected: '{user_question}'")
        date_facts = calculate_date_facts(user_question)
        print(f"[Date Interceptor] Injecting facts: {date_facts}")

        # Inject as system message right after main system prompt
        messages.insert(1, {
            "role": "system",
            "content": f"ACCURATE DATE FACTS (use these directly, do not recalculate): {date_facts}"
        })

    return messages


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

    # New conversation: start with the system prompt
    if conv_id not in conversations:
        conversations[conv_id] = [
            {
                "role" : "system",
                "content": f"""You are a helpful honest assistant.
                Today is {datetime.now().strftime('%A, %d %B %Y')}.

                CRITICAL RULES:
                - Do not show your reasoning process
                - For greetings like Hi or Hello reply with one short sentence only
                - For simple questions give short direct answers
                - Only give long detailed responses when the question genuinely requires it
                - Format in HTML only when it adds value
                - Use <p> for text, <ul><li> for lists, <table> for comparisons
                - Never include <html> <head> or <body> tags
                - Never over explain
                - Never make up facts"""
            }
        ]

    conversations[conv_id].append({
        "role":"user",
        "content": request.message
    })

    messages_to_send = build_messages_with_context(conversations[conv_id], request.message, conv_id, x_api_key)

    prompt = tokenizer.apply_chat_template(
        messages_to_send,
        tokenize=False,
        add_generation_prompt=True
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
    conversation_id: str = Form(...),
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
def get_documents(conv_id: str, x_api_key: str = Header(...)):
    """Get info about uploaded documents"""
    validate_key(x_api_key)
    return get_document_info(x_api_key, conv_id)


@app.delete("/documents/file/{filename}")
def remove_document(filename: str, conv_id: str = Header(...), x_api_key: str = Header(...)):
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
def load_conv(conv_id: str, x_api_key: str = Header(...)):
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
def delete_conv(conv_id: str, x_api_key: str = Header(...)):
    """Delete a conversation"""
    validate_key(x_api_key)
    # if conv_id in conversations:
    #     del conversations[conv_id]
    delete_conversation(x_api_key, conv_id)
    return {"status": "deleted"}

@app.put("/conversations/{conv_id}/rename")
def rename_conv(
    conv_id: str,
    body: dict,
    x_api_key: str = Header(...)
):
    """Rename a conversation"""
    validate_key(x_api_key)
    rename_conversation(x_api_key, conv_id, body.get("title", ""))
    return {"status": "renamed"}
