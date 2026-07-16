"""
Central config. Loads API keys from .env and defines shared constants.
Every other file in this project imports from here instead of
hardcoding values — so if you change a model or a folder path,
you only change it in one place.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file and loads it into the environment

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.3-70b-versatile"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHROMA_DIR = "data/chroma_store"
RAW_DOCS_DIR = "data/raw_contracts"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 8

MAX_CRITIQUE_LOOPS = 2

# LangSmith tracing is picked up automatically via environment variables
# (LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT) once .env is loaded.
# This just gives us a visible confirmation when the app starts.
import os as _os
if _os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print(f"[LangSmith] Tracing enabled → project: {_os.getenv('LANGCHAIN_PROJECT')}")
else:
    print("[LangSmith] Tracing NOT enabled — check .env")