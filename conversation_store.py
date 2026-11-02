"""
Conversation storage: one JSON file per conversation, one folder per API key.

    data/conversations/<client>/<conv_id>.json
"""

import json
import os
from datetime import datetime
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
    os.makedirs(path, exist_ok=True)
    return path

def get_conv_path(client_id: str, conv_id: str) -> str:
    """Get full path to a conversation file"""
    path = ensure_path(client_id)
    return os.path.join(path, f"{conv_id}.json")

def save_conversation(client_id: str, conv_id: str, messages: list, title: str = None):
    """Save conversation to disk; the title defaults to the first user message."""
    try:
        path = get_conv_path(client_id, conv_id)

        # The file is rewritten on every save, so keep the original created_at
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

        # ensure_ascii=False keeps Arabic text readable in the file
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=JSON_INDENT)

        print(f"[STORE] Saved conversation {conv_id}")

    except Exception as e:
        print(f"[STORE] Error saving conversation: {str(e)}")

def load_conversation(client_id: str, conv_id: str) -> Optional[dict]:
    """Load a conversation from disk, or return None if it doesn't exist."""
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
                        "message_count": len([
                            m for m in data["messages"]
                            if m["role"] != "system"
                        ])
                    })

        # Newest first (the YYYY-MM-DD HH:MM format sorts correctly as text)
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
