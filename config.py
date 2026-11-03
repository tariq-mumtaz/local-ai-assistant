"""Load settings from .env and validate them. Import before other local modules."""

import os
import sys

from dotenv import load_dotenv
from pathlib import Path

# system prompt: edit prompts/system_prompt.txt to change the system prompt
SYSTEM_PROMPT_FILE = Path(__file__).parent / os.getenv("SYSTEM_PROMPT_FILE", "prompts/system_prompt.txt")
if not SYSTEM_PROMPT_FILE.exists():
    sys.exit(f"ERROR: System prompt file not found: {SYSTEM_PROMPT_FILE}. Copy prompts/system_prompt.txt.example to prompts/system_prompt.txt and edit it.")
    
load_dotenv()   # reads .env into os.environ

def load_api_keys():
    """Read API_KEYS from .env, e.g. API_KEYS=key-one:Tariq,key-two:Client One"""
    keys = {}
    for pair in os.getenv("API_KEYS","").split(","):
        if ":" in pair:
            key,name = pair.split(":",1)
            keys[key.strip()] = name.strip()
    return keys

API_KEYS = load_api_keys()
if not API_KEYS:
    sys.exit("ERROR: No API keys found. Copy .env.example to .env and set API_KEYS.")

if "change-this-key" in API_KEYS:
    sys.exit("ERROR: Replace the example key 'change-this-key' in .env with your own key.")

MODEL_NAME = os.getenv("MODEL_NAME", "mlx-community/Qwen3-8B-4bit")

# maximum cosine distance for a chunk to consider relevent, 0 is equal identical meaning
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE","0.08"))