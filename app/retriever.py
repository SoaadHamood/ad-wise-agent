from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import requests

from app.settings import (
    EMBED_MODEL_NAME,
    MAX_ADS_PER_MATCH,
    MAX_CTX_CHARS,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    TOP_K,
)

_index = None

GLOBAL_ADS_BUDGET: int = int(os.getenv("GLOBAL_ADS_BUDGET", "50"))
RAG_KEYWORDS = os.getenv("RAG_KEYWORDS", "").strip()
RAG_KEYWORD_LIST = [k.strip().lower() for k in RAG_KEYWORDS.split(",") if k.strip()]

# HuggingFace Inference API - free, no local model download needed
_HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/{EMBED_MODEL_NAME}"


@dataclass
class RetrievalTrace:
    provider: str
    top_k: int
    namespace: str
    index_name: str
    matches: int
    categories: List[str]
    ads_used: int
    note: str = ""


def _get_embedding_via_api(text: str) -> List[float]:
    """
    Get sentence embedding via HuggingFace Inference API.
    No local model loaded — zero RAM overhead on the server.
    Set HF_TOKEN env var for higher rate limits.
    """
    hf_token = os.getenv("HF_TOKEN", "")
    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    payload = {"inputs": text, "options": {"wait_for_model": True}}

    r = requests.post(_HF_API_URL, headers=headers, json=payload, timeout=30)
    if not r.ok:
        raise RuntimeError(f"HF API error {r.status_code}: {r.text[:200]}")

    data = r.json()
    # Response shape: [[float...]] (token embeddings) or [float...] (sentence)
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], list):
            # Average token embeddings to get sentence embedding
            n = len(data)
            dim = len(data[0])
            vec = [sum(data[i][j] for i in range(n)) / n for j in range(dim)]
            return vec
        return data  # Already a flat vector
    raise RuntimeError(f"Unexpected HF response: {type(data)}")


def _normalize_vec(vec: List[float]) -> List[float]:
    magnitude = sum(x * x for x in vec) ** 0.5
    if magnitude == 0:
        return vec
    return [x / magnitude for x in vec]


def _get_index():
    global _index
    if _index is not None:
        return _index
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY.")
    from pinecone.grpc import PineconeGRPC
    pc = PineconeGRPC(api_key=api_key)
    _index = pc.Index(PINECONE_INDEX_NAME)
    return _index


def preload_model() -> None:
    """No-op — embeddings are via API now. Kept for interface compatibility."""
    import logging
    logging.getLogger("retriever").info("Embedding mode: HuggingFace Inference API (no local model) ✓")


def _normalize_metadata(md: Any) -> Dict[str, Any]:
    if md is None:
        return {}
    if isinstance(md, dict):
        return md
    try:
        return dict(md)
    except Exception:
        return {}


def _looks_like_product_title(s: str) -> bool:
    s = (s or "").strip()
    if len(s) < 6:
        return False
    return sum(1 for ch in s if ch.isalpha()) >= 3


def _split_ads_blob(blob: str) -> List[str]:
    blob = (blob or "").strip()
    if not blob:
        return []
    for delim in ("\n---\n", "\n\n", "\n", "|||", "|~|"):
        if delim in blob:
            parts = [p.strip(" \t\r\n-•") for p in blob.split(delim)]
            return [p for p in parts if _looks_like_product_title(p)]
    if blob.count(",") >= 30:
        parts = [p.strip(" \t\r\n-•") for p in blob.split(",")]
        return [p for p in parts if _looks_like_product_title(p)]
    return [blob]


def _extract_ads_from_metadata(md: Dict[str, Any]) -> List[str]:
    ads = md.get("ads")
    if isinstance(ads, list):
        return [str(a).strip() for a in ads if _looks_like_product_title(str(a))]
    ads_json = md.get("ads_json") or md.get("ads_list")
    if isinstance(ads_json, str):
        try:
            parsed = json.loads(ads_json)
            if isinstance(parsed, list):
                return [str(a).strip() for a in parsed if _looks_like_product_title(str(a))]
        except Exception:
            pass
    blob = md.get("ads_blob") or md.get("ad_text") or md.get("text") or ""
    return _split_ads_blob(str(blob))


def _keyword_filter(ads: List[str], query_text: str) -> List[str]:
    if not ads:
        return []
    q_tokens = {t for t in re.findall(r"[a-z0-9]+", query_text.lower()) if len(t) >= 3}

    def keep(a: str) -> bool:
        t = a.lower()
        if RAG_KEYWORD_LIST:
            return any(k in t for k in RAG_KEYWORD_LIST)
        return any(tok in t for tok in q_tokens)

    kept = [a for a in ads if keep(a)]
    return kept if len(kept) >= min(5, len(ads)) else ads


def _score_ad(ad: str, q_tokens: set) -> int:
    t = ad.lower()
    return sum(1 for tok in q_tokens if tok in t)


def _format_ctx(blocks: List[str]) -> str:
    return ("\n\n".join(blocks))[:MAX_CTX_CHARS]


def _local_db_path() -> str:
    return os.getenv("LOCAL_AMAZON_FTS_DB") or os.path.join("data", "amazon_ads_fts.sqlite")


def _fts_fallback(query: str, top_k: int) -> Tuple[str, RetrievalTrace]:
    path = _local_db_path()
    if not os.path.exists(path):
        return "", RetrievalTrace(
            provider="none", top_k=top_k, namespace="", index_name="",
            matches=0, categories=[], ads_used=0,
            note="No Pinecone and no local FTS DB available",
        )
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    toks = [t for t in query.lower().split() if len(t) >= 3][:12]
    if not toks:
        conn.close()
        return "", RetrievalTrace(
            provider="local_fts", top_k=top_k, namespace="", index_name=path,
            matches=0, categories=[], ads_used=0, note="Query had no usable tokens",
        )
    fts_query = " OR ".join(toks)
    fetch_k = max(50, top_k * 20)
    try:
        cur.execute("SELECT ad_text FROM ads_fts WHERE ads_fts MATCH ? LIMIT ?", (fts_query, fetch_k))
        rows = cur.fetchall()
    finally:
        conn.close()
    seen = set()
    ads: List[str] = []
    for r in rows:
        t = (r[0] or "").strip()
        if not t or t.lower() in seen:
            continue
        seen.add(t.lower())
        ads.append(t)
        if len(ads) >= top_k:
            break
    blocks = ["[Category: unknown]\n" + "\n".join(f"- {a}" for a in ads)] if ads else []
    return _format_ctx(blocks), RetrievalTrace(
        provider="local_fts", top_k=top_k, namespace="", index_name=path,
        matches=len(ads), categories=["unknown"], ads_used=len(ads),
        note="Used local SQLite FTS fallback",
    )


def _pinecone_query(index, namespace: str, vec: List[float], top_k: int, md_filter: Optional[Dict[str, Any]]):
    kwargs = {"namespace": namespace, "vector": vec, "top_k": top_k, "include_metadata": True}
    if md_filter:
        kwargs["filter"] = md_filter
    return index.query(**kwargs)


def retrieve_examples(query_text: str, category_filter: str = "") -> Tuple[str, RetrievalTrace]:
    if not os.getenv("PINECONE_API_KEY"):
        return _fts_fallback(query_text, TOP_K)

    used_filter_note = "no_filter"
    try:
        raw_vec = _get_embedding_via_api(query_text)
        vec = _normalize_vec(raw_vec)
        index = _get_index()

        filters_to_try: List[Tuple[Optional[Dict[str, Any]], str]] = [(None, "no_filter")]
        if category_filter:
            keys = ["category", "category_id", "cat", "folder", "pinecone_id"]
            filters_to_try = [({k: {"$eq": category_filter}}, f"filter:{k}") for k in keys]
            filters_to_try.append((None, "no_filter_fallback"))

        res = None
        for f, note in filters_to_try:
            res_try = _pinecone_query(index, PINECONE_NAMESPACE, vec, TOP_K, f)
            matches_try = getattr(res_try, "matches", None) or []
            if matches_try or f is None:
                res = res_try
                used_filter_note = note
                break

    except Exception as e:
        ctx, trace = _fts_fallback(query_text, TOP_K)
        trace.note = f"Retrieval failed; fallback used. Reason: {type(e).__name__}: {e}"
        return ctx, trace

    matches = getattr(res, "matches", None) or []
    q_tokens = {t for t in re.findall(r"[a-z0-9]+", query_text.lower()) if len(t) >= 3}

    blocks: List[str] = []
    cats: List[str] = []
    total_ads = 0

    for m in matches:
        if total_ads >= GLOBAL_ADS_BUDGET:
            break
        md = _normalize_metadata(m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", None))
        cat = str(md.get("category", md.get("folder", md.get("category_id", "unknown"))))
        cats.append(cat)
        ads_raw = _extract_ads_from_metadata(md)
        if not ads_raw:
            continue
        ads_raw = _keyword_filter(ads_raw, query_text)
        ads_raw = sorted(ads_raw, key=lambda a: _score_ad(a, q_tokens), reverse=True)
        seen = set()
        ads: List[str] = []
        for a in ads_raw:
            k = a.lower().strip()
            if not k or k in seen:
                continue
            seen.add(k)
            ads.append(a.strip())
            if len(ads) >= MAX_ADS_PER_MATCH:
                break
        ads = ads[:GLOBAL_ADS_BUDGET - total_ads]
        if not ads:
            continue
        total_ads += len(ads)
        blocks.append(f"[Category: {cat}]\n" + "\n".join(f"- {a}" for a in ads))

    ctx = _format_ctx(blocks)
    note = used_filter_note
    if category_filter and used_filter_note == "no_filter_fallback":
        note = f"Category filter '{category_filter}' returned no matches; fell back to unfiltered"

    return ctx, RetrievalTrace(
        provider="pinecone",
        top_k=TOP_K,
        namespace=PINECONE_NAMESPACE,
        index_name=PINECONE_INDEX_NAME,
        matches=len(matches),
        categories=sorted(set(c for c in cats if c)),
        ads_used=total_ads,
        note=note,
    )
