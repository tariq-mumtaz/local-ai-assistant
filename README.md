# Local AI Assistant

A private AI assistant that runs entirely on your own machine — no cloud APIs,
so your conversations and documents never leave your computer.

- Chat in **English and Arabic**
- Upload PDFs and ask questions about them (RAG)
- Replies stream word by word; conversations are saved locally

**Status: work in progress.** Currently runs on Apple Silicon Macs (M1–M4).
Security fixes and Windows/Linux support are in progress.

## Built with

FastAPI · MLX (Qwen3-8B) · ChromaDB · sentence-transformers · PyMuPDF

## Run it

Requires an Apple Silicon Mac and Python 3.11+.

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env        # then set your own API_KEYS in .env
    uvicorn api:app --host 127.0.0.1 --port 8000

Open http://localhost:8000 and enter one of your API keys.
The first start downloads the model (about 5 GB).
