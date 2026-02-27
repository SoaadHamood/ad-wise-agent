from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Core limits
# -------------------------
MAX_PROMPT_CHARS: int = int(os.getenv("MAX_PROMPT_CHARS", "4000"))
MAX_CTX_CHARS: int = int(os.getenv("MAX_CTX_CHARS", "7000"))

# -------------------------
# Pinecone (RAG)
# -------------------------
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "amazon-ads-index")
PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "amazon_ads")
TOP_K: int = int(os.getenv("TOP_K", "5"))
MAX_ADS_PER_MATCH: int = int(os.getenv("MAX_ADS_PER_MATCH", "25"))

# Sentence-transformers model used to embed the user query
EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

# -------------------------
# LLM provider
# -------------------------
LLM_MODEL: str = os.getenv("LLM_MODEL", os.getenv("LLMOD_MODEL", "reasoning"))
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", os.getenv("LLMOD_BASE_URL", "")).strip()

# Optional: enable a second "repair" call if the model output format is invalid
ENABLE_REPAIR: bool = os.getenv("ENABLE_REPAIR", "false").strip().lower() in {"1", "true", "yes"}