from __future__ import annotations

import os
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import requests

DEFAULT_LLMOD_API_BASE = "https://api.llmod.ai"  # official API


def _get_api_key() -> str:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("LLMOD_API_KEY")
    if not api_key:
        raise RuntimeError("Missing LLM_API_KEY (or LLMOD_API_KEY). Put it in env or .env")
    return api_key


def _pick_base_url() -> str:
    """
    Pick the API base URL.
    If the user mistakenly provides the UI host (app.llmod.ai/...), ignore it.
    """
    env_base = (os.getenv("LLM_BASE_URL") or os.getenv("LLMOD_BASE_URL") or "").strip()
    if not env_base:
        return DEFAULT_LLMOD_API_BASE

    parsed = urlparse(env_base)
    if parsed.scheme and parsed.netloc:
        origin = f"{parsed.scheme}://{parsed.netloc}"
    else:
        origin = env_base.rstrip("/")

    if "app.llmod.ai" in origin:
        return DEFAULT_LLMOD_API_BASE

    return origin.rstrip("/")


def _chat_completions_url(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def llm_generate_ad(system_msg: str, user_msg: str) -> Tuple[str, Dict[str, Any]]:
    api_key = _get_api_key()
    base = _pick_base_url()
    url = _chat_completions_url(base)

    model_name = os.getenv("LLM_MODEL") or os.getenv("LLMOD_MODEL") or "reasoning"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if not r.ok:
        raise RuntimeError(
            f"LLM call failed: status={r.status_code} url={url} model={model_name} body={r.text[:2000]}"
        )

    data = r.json()
    text = data["choices"][0]["message"]["content"]

    meta = {
        "used_url": url,
        "model": model_name,
        "text_preview": (text or "")[:1200],
        "raw_usage": data.get("usage"),
    }

    return text, meta