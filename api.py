# FastAPI: creates the app itself.
# HTTPException: sends an error back to the browser, e.g. “401 Invalid API Key”.
# Header: reads values from request headers, such as x-api-key.
# UploadFile, File, Form: receive uploaded files and form fields (using for document upload).

from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form

# StaticFiles: serves whole folders as-is (scripts/ and styles/).
# FileResponse: sends back one file (chat.html).
# When you open http://localhost:8000, FileResponse chat.html to the browser,

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# These come from rag.py file. They’re the document functions: 
# save a PDF, find the parts that match a question, check whether documents exist, 
# list them, and delete one or all. 
# Importing rag also runs its top lines, 
# which is why “Loading embedding model...” and “ChromaDB ready!” appear first when the server starts.

# from rag import store_document, retrieve_relevant_chunks, has_documents, get_document_info, clear_documents
from rag import (
    store_document,
    retrieve_relevant_chunks,
    has_documents,
    get_document_info,
    delete_document,
    clear_all_documents
)
from conversation_store import (
    save_conversation,
    load_conversation,
    list_conversations,
    delete_conversation,
    rename_conversation
)

# datetime: dates and times, e.g. getting today’s date.
# timedelta: a length of time, e.g. “+1 day” for tomorrow.

from datetime import datetime,timedelta

# from fastapi import FastAPI, HTTPException, Header

# Sends a reply piece by piece instead of all at once. 
# This is what makes the answer appear word by word in the chat.
from fastapi.responses import StreamingResponse

# CORS controls which websites may call your API from a browser.
from fastapi.middleware.cors import CORSMiddleware

# BaseModel: describes the shape of the data a request must contain (e.g. message must be text). 
# FastAPI checks incoming requests against it automatically.
# Optional: marks a field as allowed to be missing, like conversation_id on a new chat.

from pydantic import BaseModel
from typing import Optional

# MLX: Apple released it as open source in December 2023. 
# It’s built specifically for M-series chips like your M3, 
# which share one pool of memory between the CPU and GPU, so models run fast without copying data back and forth.
# That’s also why it works only on Apple Silicon Macs.

# The AI part, from MLX:
# load: loads the model into memory.
# generate: produces a full reply in one go.
# stream_generate: produces the reply one piece at a time.
# make_sampler: sets how random the word choice is (temperature).
# make_logits_processors: adds rules such as “don’t repeat yourself” (repetition penalty).

from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

# uuid: creates unique IDs for new conversations, like 5c39c2ca-dd2d-....
# json: converts data to and from JSON text.
# re: regular expressions, used to search and clean text.
# os: works with files and folders.
# Path: an easier way to build file paths.
# sys.exit("message") stops the program, prints the message, and reports failure. 
# Uvicorn then stops too, and the message is the last thing in Terminal, so it’s impossible to miss.

import uuid
import json
import re
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()   # reads .env into os.environ


#API Keys
def load_api_keys():
    """
    Read API_KEYS from .env, eg API_KEYS=my-secret-key-123:Tariq,client-key-456:Client One
    """
    keys = {}
    for pair in os.getenv("API_KEYS",{}).split(","):
        if ":" in pair:
            key,name = pair.split(":",1)
            keys[key.strip()] = name.strip()
    return keys

API_KEYS = load_api_keys()
if not API_KEYS:
    sys.exit("ERROR: No API keys found. Copy .env.example to .env and set API_KEYS.")

if "change-this-key" in API_KEYS:
    sys.exit("ERROR: Replace the example key 'change-this-key' in .env with your own key.")



# Do not need to login in to Hugging Face, it will auto read HF_TOKEN from env and download the model
# Creates your web application and names it app. Every endpoint later attaches to it with @app.get(...), 
# @app.post(...) and so on.
# This is the app in your start command: uvicorn api:app means “in the file api.py, run the object called app”.
app = FastAPI(title="My AI API")

#Allow browser request from any origin
# CORS (Cross-Origin Resource Sharing) is how a server says “these other origins are allowed”. 
# Before a cross-origin request, the browser first asks the server for permission with an OPTIONS request, 
# called a “preflight”. Those are the OPTIONS /conversations and OPTIONS /chat/stream lines in our log yesterday. 
# They mean your page and API were on different origins, 
# most likely because uvicorn prints 127.0.0.1:8000 while the original chat.js calls localhost:8000. 
# CORS allowed it, so everything worked.

# The settings:

# app.add_middleware(...): middleware is code that runs on every request before your endpoints. 
# Here it adds the CORS permission headers.
# allow_origins=["*"]: * means “anyone”: any website may call this API.
# allow_methods=["*"]: any request type: GET, POST, PUT, DELETE...
# allow_headers=["*"]: any header, including your custom x-api-key.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# Builds the path to Hugging Face’s download folder:
# Path.home() is your home folder, /Users/username.
# With Path objects, / joins path parts, the same job as os.path.join in conversation_store.py, 
# just a newer style. Here it isn’t division.
# Result: /Users/username/.cache/huggingface/hub

#load model once when API starts - not on every request
def get_model_cache_path(model_name: str) -> Path:
    """Check if model is already cached"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    # Convert model name to cache folder format
    # Hugging Face names its folders in a fixed pattern: / becomes --, with models-- in front.
    # mlx-community/Qwen3-8B-4bit becomes .../hub/models--mlx-community--Qwen3-8B-4bit
    
    safe_name = model_name.replace("/", "--")
    return cache_dir / f"models--{safe_name}"

MODEL_NAME = os.getenv("MODEL_NAME","mlx-community/Qwen3-8B-4bit")


# Check if already downloaded
cache_path = get_model_cache_path(MODEL_NAME)
if cache_path.exists():
    print(f"Model found in cache — loading...")
else:
    print(f"Model not found in cache")
    print(f"Downloading {MODEL_NAME}...")
    print(f"This is a one-time download of approximately 5GB")
    print(f"Please wait...")


# MLX’s load downloads the model if needed, reads it into memory, and returns two things, 
# which are unpacked into two names
# model: the neural network itself, billions of numbers that predict the next piece of text.
# tokenizer: converts between text and tokens. Models don’t read letters; 
# they read numbered word-pieces. “Hello world” might become [9707, 1879]. 
# The tokenizer converts your messages into numbers before generation, 
# and the model’s numbers back into text after. 
# It also knows the model’s chat template: the special format the model expects 
# for system, user and assistant messages.

print("Loading model into memory...")
model, tokenizer = load(MODEL_NAME)

# top_p=0.8: only consider the most likely tokens that together make up 80% of the probability, 
# and ignore the long tail of unlikely ones. This keeps answers sensible even with a bit of randomness.
# top_k=20: never consider more than the top 20 tokens.
# min_p=0.0: an extra filter, switched off.
# Keep sampler_zero at 0. The greedy-decoding caution is about normal answers in thinking mode. 
# Pulling a date out of a sentence is a tiny exact task, where being predictable is what you want.
# repetition_penalty=1.05: just a light touch. You could also remove it entirely and only add it back if you see repetition.

# Normal chat — Qwen3 recommended settings for non-thinking mode
sampler = make_sampler(temp=0.7, top_p=0.8, top_k=20, min_p=0.0)

# Date extraction — keep fully predictable
sampler_zero = make_sampler(temp=0.0)

# Gentle repetition control (1.3 was tuned for Phi-3's runaway problem)
logits_processors = make_logits_processors(
    repetition_penalty=1.05,
    repetition_context_size=20
)

print("Model ready!")

# conversation memory

conversations = {}

#request and response shapes
class ChatRequest(BaseModel):
    message:str
    conversation_id:Optional[str] = None

def validate_key(api_key:str):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return API_KEYS[api_key]

def clean_response(response: str) -> str:
    # Remove think tags and their content
    # First line: finds <think>, then anything, then </think>, and replaces it all with nothing ('').
    # [\s\S] means “any character, including line breaks”.
    # *? means “as little as possible”, so it stops at the first </think>.
    # Second line: a backup for when the reply was cut off while still thinking, 
    # so there’s a <think> but no </think>. Here * (without ?) takes everything to the end.

    response = re.sub(r'<think>[\s\S]*?</think>', '', response)
    response = re.sub(r'<think>[\s\S]*', '', response)  # unclosed think tag
    
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

    # This line matters. .copy() makes a new list. Without it, 
    # messages would just be another name for the same list, the one stored in conversations. 
    # Then every document excerpt and date fact added below would be saved permanently in the 
    # chat history and sent again with every future message, piling up turn after turn.
    # With the copy, the additions exist only for this one reply. The saved history stays clean.
    
    messages = conversations_list.copy()

    # RAG — inject relevant document chunks if documents uploaded
    if client_id and has_documents(client_id, conv_id):
        relevant = retrieve_relevant_chunks(client_id, user_question, conv_id)
        if relevant:
            print(f"[RAG] Injecting relevant chunks for: '{user_question}'")
            messages.insert(1, {
                "role": "system",
                "content": relevant
            })

    # messages.insert(1, ...) puts it at position 1: 
    # right after the main system prompt (position 0) and before the conversation history:

    # AI models are bad at calendar maths (“What day is 30 April 2026?”), 
    # so for date questions Python works out the correct answer (Part 6) and gives it to the model as a fact.
    # This also inserts at position 1, so if both apply, 
    # the date facts end up in front of the document excerpts:
    # Date interception

    if is_date_question(user_question):
        date_facts = calculate_date_facts(user_question)
        messages.insert(1, {
            "role": "system",
            "content": f"ACCURATE DATE FACTS (use directly): {date_facts}"
        })

    return messages

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


# Counts the Arabic letters in the text:
# Every character has a number (its Unicode code). Arabic letters are in the range 0600 to 06FF (hexadecimal). '\u0600' is the first character in that range, and '\u06FF' the last.
# '\u0600' <= c <= '\u06FF' is true if character c falls in that range. Python allows this chained comparison, meaning “between”.
# sum(1 for c in text if ...) adds 1 for every character that passes, i.e. counts them. It’s like a list comprehension, but with round brackets, so it counts without building a list.

def detect_language(text: str) -> str:
    """Detect if text is Arabic or English"""
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0:
        return "english"
    return "arabic" if (arabic_chars / total_chars) > 0.3 else "english"

# Arabic text often uses Arabic-Indic digits (٠١٢٣...). 
# The regex below only understands 0–9, so this converts them first.
# zip pairs the two strings character by character: ('٠','0'), ('١','1'), ... ('٩','9'), 
# the same zip as in rag.py.
# Each round replaces one Arabic digit everywhere in the text.

def convert_arabic_numerals(text: str) -> str:
    """Convert Arabic-Indic numerals ٠١٢٣ to Western 0123"""
    arabic_indic = "٠١٢٣٤٥٦٧٨٩"
    western = "0123456789"
    for a, w in zip(arabic_indic, western):
        text = text.replace(a, w)
    return text

def is_date_question(question: str) -> bool:
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

    # Also check if question contains a year number — strong signal
    # Looks for a year from 2000–2099:
    # 20\d{2}: “20” followed by exactly 2 digits.
    # \b: a word boundary, the edge of a word. It makes sure 2026 matches but not part of 120265.
    # re.search returns a match or None, and bool(...) turns that into True/False.
    
    has_year = bool(re.search(r'\b20\d{2}\b', question))
    # any(...) is True if at least one keyword appears in the question. .lower() 
    # makes it ignore capitals (“What Day” matches “what day”).
    has_date_word = any(kw in question.lower() for kw in keywords)

    return has_date_word or has_year  # NEW: year alone triggers interception


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

    # Pattern: "30th April 2026" or "30 April 2026"
    # (\d{1,2}): group 1: the day, 1 or 2 digits. Round brackets capture a part so you can get it back later.
    # (?:st|nd|rd|th)?: an optional ending like “th”. (?:...) groups without capturing, and the final ? makes it optional.
    # \s+: one or more spaces.
    # (january|...|december): group 2: the month name.
    # (\d{4}): group 3: a 4-digit year.

    pattern = r'(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})'
    match = re.search(pattern, question, re.IGNORECASE)

    if match:
        try:
            day = int(match.group(1))
            month = months_en[match.group(2).lower()]
            year = int(match.group(3))
            return datetime(year, month, day)
        except ValueError:
            pass

    # Pattern: "April 30 2026" or "April 30th 2026"
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

    # Pattern: dd/mm/yyyy or dd-mm-yyyy
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

    # Try regex first — fast and always correct
    regex_result = extract_date_with_regex(question)
    if regex_result:
        print(f"[Date Extractor] Regex found: {regex_result.strftime('%d/%m/%Y')}")
        return regex_result

    # Fall back to model for complex phrasing
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
    # Turns the messages into the exact text format Qwen3 expects
    extraction_prompt = tokenizer.apply_chat_template(
        extraction_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Runs the model:
    # max_tokens=15: a date like 30/04/2026 is only a few tokens, so this keeps the call short.
    # sampler=sampler_zero: temperature 0, so the same question always gives the same answer.

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
    # strptime is the opposite of strftime: it reads text in a given format and turns it into a date, 
    # so "30/04/2026" becomes a real date. If the model replied with anything else 
    # (“The date is 30 April”), it fails with ValueError and returns None.
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
    # today.strftime('%A, %d %B %Y') formats the date as text:
    # %A gives Wednesday (day name)
    # , is just a comma
    # %d gives 28 (day)
    # %B gives October (month name)
    # %Y gives 2026 (year)
    # Result: "Wednesday, 28 October 2026"
    # The f"..." string puts that into the sentence: "Today is Wednesday, 28 October 2026."
    # result.append(...) adds the sentence to the result list.

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


app.mount("/scripts", StaticFiles(directory="scripts"), name="scripts")
app.mount("/styles", StaticFiles(directory="styles"), name="styles")

@app.get("/")
def root():
    return FileResponse("chat.html")



# x_api_key: str = Header(...) reads the API key from the request headers:
# Header(...) means “take this value from a header”.
# The ... (three dots, called Ellipsis) means required. 
# If the header is missing, FastAPI rejects the request automatically with a 422 error.
# FastAPI converts the name: x_api_key in Python means the header x-api-key. 
# (Hyphens aren’t allowed in Python names, so underscores are used and swapped automatically.)
# return {"status": "healthy"}: 
# FastAPI turns a returned dictionary into JSON automatically, so the browser receives {"status": "healthy"}.

@app.get("/health")
def health(x_api_key: str = Header(...)):
    validate_key(x_api_key)
    return {"status": "healthy"}


# @app.post: a POST request, because the browser is sending something new (a message).
# request: ChatRequest: FastAPI reads the JSON body and checks it against ChatRequest, 
# so request.message and request.conversation_id are ready to use.
# validate_key stops the request with 401 if the key is wrong. client_name is stored but never used.

@app.post("/chat/stream")
def chat_stream(request:ChatRequest,x_api_key: str = Header(...)):
    #validate API key
    client_name = validate_key(x_api_key)

    #get or create conversation id
    conv_id = request.conversation_id or str(uuid.uuid4())

    # ← ADD THIS BLOCK HERE
    if conv_id not in conversations:
        saved = load_conversation(x_api_key, conv_id)
        if saved:
            conversations[conv_id] = saved["messages"]
            print(f"[STORE] Restored conversation {conv_id} from disk")
    # ← END OF ADDED BLOCK

    
    #create new conversation with system prompt
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

    #add user message to conversation
    conversations[conv_id].append({
        "role":"user",
        "content": request.message  
    })

    # CHANGED: use build_messages_with_date before building prompt
    messages_to_send = build_messages_with_context(conversations[conv_id], request.message, conv_id, x_api_key)

        
    #build prompt
    prompt = tokenizer.apply_chat_template(
        messages_to_send,
        tokenize=False,
        add_generation_prompt=True
    )

    # A function inside a function. It can use conv_id, prompt and so on from the outer function. 
    # full_response collects the whole reply for saving later.
    # This is a generator function: it uses yield instead of return.
    # return gives back one result and ends the function.
    # yield gives back one piece, pauses, and continues from the same spot when the next piece is wanted.

    def generate_stream():
        full_response = ""

        #send conversation id first
        # The first piece sent is the conversation ID, so a brand-new chat learns its ID straight away.
        # The format is Server-Sent Events (SSE), a standard way for a server to push messages to a browser:
        # Each message starts with data: .
        # It ends with two line breaks, \n\n, which marks the end of one message.
        # json.dumps(...) turns the dictionary into JSON text (dump = to a file, dumps = to a string).
        # So the browser receives:
    
        yield f"data: {json.dumps({'type': 'conv_id', 'conversation_id': conv_id})}\n\n"

        #sream token one by one
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
            # Send token to client
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        
        #clean final response and save to history
        full_response = full_response.strip()
        conversations[conv_id].append({
            "role":"assistant",
            "content": full_response
        })
        # Save to disk
        save_conversation(x_api_key, conv_id, conversations[conv_id])
        
        #signal stream is end
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    # StreamingResponse(generate_stream(), ...): calling generate_stream() doesn’t run it yet; 
    # it creates the generator. StreamingResponse then pulls pieces from it one by one 
    # and sends each to the browser immediately.
    # media_type="text/event-stream" tells the browser “this is an SSE stream”.
    # "Cache-Control": "no-cache": don’t store this response; it’s live.
    # "X-Accel-Buffering": "no": a message for nginx, 
    # a common web server placed in front of apps on cloud servers. Without it, 
    # nginx collects the whole reply before passing it on, and the word-by-word effect is lost. 
    # It does nothing on your Mac but helps on a server, so it’s good that it’s there.


    return StreamingResponse(generate_stream(), media_type="text/event-stream",headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        })



# Uploads don’t send JSON. They send form data (multipart), the same format as a file field in an HTML form:
# file: UploadFile = File(...): the uploaded file (required, ...).
# conversation_id: str = Form(...): a normal text field sent alongside the file.
# x_api_key: from the header, as always.
# This is why the python-multipart package is needed: FastAPI uses it to read form data.
# async def instead of def: an async function can pause while waiting (e.g. for a file to arrive) 
# and let the server handle other requests meanwhile. It’s needed here because of await below

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
    x_api_key: str = Header(...)
):
    client_name = validate_key(x_api_key)

    # Only accept PDFs for now
    # if not file.filename.endswith(".pdf"):
    #     raise HTTPException(
    #         status_code=400,
    #         detail="Only PDF files supported currently"
    #     )

    # Read file bytes
    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Store and index the document
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