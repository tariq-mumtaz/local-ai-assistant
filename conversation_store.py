# Converts between Python data and JSON text.
import json
# Works with files and folders: building paths, creating folders, checking whether a file exists, deleting.
import os
# Gets the current date and time for the created_at and updated_at timestamps.
from datetime import datetime
# Type-hint labels. Optional[dict] means “a dictionary, or None”.
from typing import List, Optional

STORE_PATH = "./data/conversations"
JSON_INDENT = 4

def get_client_path(client_id: str) -> str:
    """Get path for client's conversations folder"""
    safe_id = client_id.replace("/", "-").replace(":", "-")
    return os.path.join(STORE_PATH, safe_id)

def ensure_path(client_id: str):
    """Create folders if they don't exist"""
    path = get_client_path(client_id)
    # os.makedirs creates the folder, plus any missing parent folders. 
    # So on a fresh install it creates data, then conversations, then dev-key-999 in one step.
    # exist_ok=True means “if it already exists, that’s fine; don’t raise an error”. 
    # Without it, the second save would crash with “folder already exists”.

    os.makedirs(path, exist_ok=True)
    return path

def get_conv_path(client_id: str, conv_id: str) -> str:
    """Get full path to a conversation file"""
    path = ensure_path(client_id)
    return os.path.join(path, f"{conv_id}.json")

def save_conversation(client_id: str, conv_id: str, messages: list, title: str = None):
    """
    Save conversation to disk
    client_id and conv_id: which file to write.
    messages: the whole conversation as a list, e.g. [{"role": "system", ...}, 
    {"role": "user", "content": "Hi"}, 
    {"role": "assistant", ...}].
    title: str = None: an optional title. If not given, one is created automatically.
    """
    try:
        path = get_conv_path(client_id, conv_id)

        # Check if file exists to preserve created_at
        # os.path.exists(path) is True if the file is already there, i.e. not the first message.
        # with open(path, 'r') as f: opens the file for reading. The with block automatically closes the file when the block ends, even if an error happens. It’s the standard, safe way to open files in Python.
        # json.load(f) reads the JSON text and turns it into a Python dictionary.
        # existing.get("created_at", ...) gets the created_at value. .get(key, default) returns the default instead of crashing if the key is missing. Here the default is “now”.


        # Code	Meaning	Example
        # %Y	4-digit year	2026
        # %m	month	10
        # %d	day	28
        # %H	hour (24h)	10
        # %M	minute	05
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                existing = json.load(f)
            created_at = existing.get("created_at", datetime.now().strftime('%Y-%m-%d %H:%M'))
        else:
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M')

        # Auto generate title from first user message if not provided
        if not title:
            for msg in messages:
                if msg["role"] == "user":
                    title = msg["content"][:50]
                    if len(msg["content"]) > 50:
                        title += "..."
                    break
            if not title:
                title = "New Conversation"

        data = {
            "conv_id": conv_id,
            "title": title,
            "created_at": created_at,
            "updated_at": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "messages": messages
        }

        # open(path, 'w') opens for writing. 'w' erases the file first and writes from scratch, or 
        # creates it if it doesn’t exist.
        # json.dump converts the dictionary to JSON text and writes it to the file. 
        # Note: json.load reads, json.dump writes.
        # ensure_ascii=False writes Arabic and other characters as they are. 
        # Without it, مرحبا would be stored as unreadable codes like \u0645\u0631.... 
        # It still works either way, but the file is hard to read.
        # indent=2 spreads the JSON over several lines with 2-space indents, 
        # which is why your JSON file was nicely laid out. Without it, everything would be on one long line.

        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=JSON_INDENT)

        print(f"[STORE] Saved conversation {conv_id}")

    except Exception as e:
        print(f"[STORE] Error saving conversation: {str(e)}")

def load_conversation(client_id: str, conv_id: str) -> Optional[dict]:
    """
    Load a conversation from disk
    -> Optional[dict] means it returns either the conversation as a dictionary, or None if it doesn’t exist.
    """
    try:
        path = get_conv_path(client_id, conv_id)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[STORE] Error loading conversation: {str(e)}")
        return None

def list_conversations(client_id: str) -> List[dict]:
    """List all conversations for a client sorted by latest first"""
    try:
        path = get_client_path(client_id)
        if not os.path.exists(path):
            return []

        # os.listdir(path) gives the names of everything in the folder.
        # .endswith('.json') skips anything that isn’t a chat file

        convs = []
        for filename in os.listdir(path):
            if filename.endswith('.json'):
                filepath = os.path.join(path, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    convs.append({
                        "conv_id": data["conv_id"],
                        "title": data.get("title", "Conversation"),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),

                        # m for m in data["messages"]: go through every message.
                        # if m["role"] != "system": only keep it if it isn’t a system message (!= means “not equal”).
                        # len([...]): count what’s left.

                        "message_count": len([
                            m for m in data["messages"]
                            if m["role"] != "system"
                        ])
                    })

        # Sorts the list so the most recently used chat is at the top:
        # .sort() sorts the list in place.
        # key= tells it what to sort by. Here: each chat’s updated_at.
        # lambda x: x["updated_at"] is a tiny unnamed function. It means the same as:
        # Sort by updated_at latest first

        convs.sort(key=lambda x: x["updated_at"], reverse=True)
        return convs

    except Exception as e:
        print(f"[STORE] Error listing conversations: {str(e)}")
        return []

def delete_conversation(client_id: str, conv_id: str) -> bool:
    """Delete a conversation from disk"""
    try:
        path = get_conv_path(client_id, conv_id)
        if os.path.exists(path):
            os.remove(path)
            print(f"[STORE] Deleted conversation {conv_id}")
            return True
        return False
    except Exception as e:
        print(f"[STORE] Error deleting conversation: {str(e)}")
        return False

def rename_conversation(client_id: str, conv_id: str, new_title: str) -> bool:
    """Rename a conversation"""
    try:
        conv = load_conversation(client_id, conv_id)
        if not conv:
            return False
        conv["title"] = new_title
        path = get_conv_path(client_id, conv_id)
        with open(path, 'w') as f:
            json.dump(conv, f, ensure_ascii=False, indent=JSON_INDENT)
        return True
    except Exception as e:
        print(f"[STORE] Error renaming: {str(e)}")
        return False