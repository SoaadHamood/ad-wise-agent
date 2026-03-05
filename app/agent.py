from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from app.retriever import retrieve_examples
from app.llm_client import llm_generate_ad
from app.settings import ENABLE_REPAIR, MAX_PROMPT_CHARS


_MODE_FULL = "full"
_MODE_HEADLINE = "headline"
_MODE_KEYWORDS5 = "keywords5"


def _detect_mode(prompt: str) -> str:
    p = (prompt or "").lower()

    # Wizard structured prompt contains "Task:"
    if "task:" in p:
        if ("write only" in p and "headline" in p) or ("headline only" in p):
            return _MODE_HEADLINE
        if ("5 keywords" in p) or ("five keywords" in p) or ("keywords that must be included" in p):
            return _MODE_KEYWORDS5
        return _MODE_FULL

    # Free-form fallback
    if "headline only" in p or ("write only" in p and "headline" in p):
        return _MODE_HEADLINE
    if "5 keywords" in p or "five keywords" in p:
        return _MODE_KEYWORDS5

    return _MODE_FULL


_FULL_FORMAT_REGEX = re.compile(
    r"^Headline:\s[^\n]+\n"
    r"Bullets:\n"
    r"(?:-\s[^\n]+\n){5}"
    r"Short description:\s[^\n]+\n"
    r"Keywords:\s[^\n]+\n"
    r"Publishing tips:\s[^\n]+$"
)
_HEADLINE_ONLY_REGEX = re.compile(r"^Headline:\s[^\n]{3,200}$")


def _validate_keywords5_line(text: str) -> bool:
    t = (text or "").strip()
    if not t.lower().startswith("keywords:"):
        return False
    rest = t.split(":", 1)[1].strip()
    if not rest:
        return False
    parts = [p.strip() for p in rest.split(",") if p.strip()]
    if len(parts) != 5:
        return False
    # avoid 1-char junk
    if any(len(p) < 2 for p in parts):
        return False
    return True


def _is_valid_format(text: str, mode: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if mode == _MODE_HEADLINE:
        return bool(_HEADLINE_ONLY_REGEX.fullmatch(t))
    if mode == _MODE_KEYWORDS5:
        return _validate_keywords5_line(t)
    return bool(_FULL_FORMAT_REGEX.fullmatch(t))


_STOPWORDS = {
    "a","an","the","and","or","for","to","of","in","on","with","without","by",
    "include","includes","including","style","amazon","ad","write","bullet","bullets",
    "cta","benefits","your","now","next","new","please","make","create",
    "this","that","these","those","from","into","over","under","best","top",
    "high","quality","premium","great"
}

# Intent hints for free text (wizard prompts skip this guard)
_INTENT_HINTS = {"write", "generate", "create", "headline", "keywords", "ad", "listing", "product"}
_PRODUCT_HINTS = {"bottle","shoe","watch","backpack","headphones","laptop","charger","camera","mug","cup","speaker","mouse","keyboard","skincare","perfume","dress"}

MIN_PROMPT_CHARS_FOR_AGENT = 12
MIN_TOKENS_FOR_AGENT = 3


def _rewrite_query(prompt: str) -> str:
    category_part = ""
    product_part = ""
    for line in (prompt or "").splitlines():
        if line.startswith("Category:"):
            category_part = line.split(":", 1)[1].strip()
        elif line.startswith("Product:"):
            product_part = line.split(":", 1)[1].strip()

    text = f"category {category_part} {product_part}".lower() if (category_part and product_part) else (prompt or "").lower()
    toks = re.findall(r"[a-z0-9]+", text)
    toks = [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]
    return " ".join(toks[:12]).strip() or (prompt or "")


def _condense_ctx(prompt: str, ctx: str, max_lines: int = 10) -> str:
    if not ctx:
        return ""
    lines = []
    for ln in ctx.splitlines():
        ln = ln.strip()
        if ln.startswith("- "):
            s = ln[2:].strip()
            if s:
                lines.append(s)

    if not lines:
        return ctx[:1200]

    q = set(re.findall(r"[a-z0-9]+", (prompt or "").lower()))
    q = {t for t in q if len(t) >= 3 and t not in _STOPWORDS}

    def score(s: str) -> int:
        t = s.lower()
        return sum(1 for tok in q if tok in t)

    seen = set()
    ranked = []
    for s in sorted(lines, key=score, reverse=True):
        k = s.lower().strip()
        if not k or k in seen:
            continue
        seen.add(k)
        ranked.append(s)
        if len(ranked) >= max_lines:
            break

    return "\n".join(f"- {s}" for s in ranked)


def _extract_allowed_terms(condensed_ctx: str, max_terms: int = 12) -> List[str]:
    if not condensed_ctx:
        return []
    toks = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", condensed_ctx.lower())
    toks = [t for t in toks if t not in _STOPWORDS]
    common = [w for w, c in Counter(toks).most_common(80) if c >= 2]
    return common[:max_terms]


def _build_messages(prompt: str, condensed_ctx: str, allowed_terms: List[str], mode: str) -> Tuple[str, str]:
    system_msg = (
        "You are Ad-Wise, an expert performance copywriter. "
        "Write high-converting ad copy using the provided inspiration examples. "
        "Be concise, avoid fluff, and follow user constraints strictly."
    )

    allowed_block = ""
    if allowed_terms:
        allowed_block = (
            "ALLOWED CLAIM TERMS (use only if they appear in inspiration or user request): "
            + ", ".join(allowed_terms)
            + "\n\n"
        )

    if mode == _MODE_HEADLINE:
        output_format = (
            "OUTPUT FORMAT (MUST be EXACT — return ONLY one line):\n"
            "Headline: ...\n\n"
            "CRITICAL RULES:\n"
            "- Output ONLY the single line starting with 'Headline: '\n"
            "- Do NOT add any other section\n"
            "- Keep the entire line <= 160 characters\n"
            "- Only use features from INSPIRATION EXAMPLES or USER REQUEST\n"
        )
    elif mode == _MODE_KEYWORDS5:
        output_format = (
            "OUTPUT FORMAT (MUST be EXACT — return ONLY one line):\n"
            "Keywords: k1, k2, k3, k4, k5\n\n"
            "CRITICAL RULES:\n"
            "- Output ONLY the single line starting with 'Keywords: '\n"
            "- Return EXACTLY 5 comma-separated keywords/phrases\n"
            "- These MUST be suitable to include in the headline\n"
            "- Do NOT add Headline, Bullets, Short description, or any other section\n"
            "- Only use terms relevant to the product and category\n"
        )
    else:
        output_format = (
            "OUTPUT FORMAT (MUST be EXACT, no extra text):\n"
            "Headline: ...\n"
            "Bullets:\n"
            "- ...\n"
            "- ...\n"
            "- ...\n"
            "- ...\n"
            "- ...\n"
            "Short description: ...\n"
            "Keywords: k1, k2, ...\n"
            "Publishing tips: ...\n\n"
            "RULES:\n"
            "1) Only mention features that appear in INSPIRATION EXAMPLES or explicitly in the USER REQUEST.\n"
            "2) Do NOT invent numeric specs unless user provides them.\n"
            "3) Keep Headline <= 160 characters.\n"
            "4) Exactly 5 bullet lines, each starting with '- '.\n"
            "5) Keywords: 8-15 comma-separated keywords; include long-tail phrases.\n"
            "6) Publishing tips: 2-3 short actionable tips, no emojis.\n"
            "7) Keep each section on a single line.\n"
        )

    user_msg = (
        f"USER REQUEST:\n{prompt}\n\n"
        f"{allowed_block}"
        "INSPIRATION EXAMPLES (most relevant excerpts):\n"
        f"{condensed_ctx if condensed_ctx else '(none)'}\n\n"
        f"{output_format}"
    )

    return system_msg, user_msg


def _repair_format(product_prompt: str, draft: str, mode: str) -> Tuple[str, Dict[str, Any]]:
    system_msg = (
        "You are a formatter. Rewrite the draft into the exact required format. "
        "Do NOT add new facts. Return ONLY the formatted output."
    )

    if mode == _MODE_HEADLINE:
        required = "Headline: ... (single line only)"
    elif mode == _MODE_KEYWORDS5:
        required = "Keywords: k1, k2, k3, k4, k5 (single line only, exactly 5)"
    else:
        required = (
            "Headline: ...\n"
            "Bullets:\n"
            "- ...\n"
            "- ...\n"
            "- ...\n"
            "- ...\n"
            "- ...\n"
            "Short description: ...\n"
            "Keywords: ...\n"
            "Publishing tips: ..."
        )

    user_msg = (
        "Required format:\n"
        f"{required}\n\n"
        f"User request:\n{product_prompt}\n\n"
        f"Draft to fix:\n{draft}"
    )

    return llm_generate_ad(system_msg, user_msg)


def _should_clarify(prompt: str) -> Tuple[bool, str]:
    p = (prompt or "").strip()
    if not p:
        return True, "empty"
    if len(p) < MIN_PROMPT_CHARS_FOR_AGENT:
        return True, "too_short"

    p_l = p.lower()
    toks = re.findall(r"[a-z0-9]+", p_l)
    meaningful = [t for t in toks if len(t) >= 2]

    has_intent = any(h in p_l for h in _INTENT_HINTS)
    has_product = any(h in p_l for h in _PRODUCT_HINTS) or ("product:" in p_l)

    if has_intent and not has_product:
        return True, "missing_product"

    if len(meaningful) < MIN_TOKENS_FOR_AGENT:
        return (False, "") if has_product else (True, "too_few_tokens")

    if not has_intent and not has_product:
        return True, "no_intent_or_product"

    return False, ""


def _clarification_response(reason: str = "") -> str:
    if reason == "missing_product":
        return (
            "Sure — what product should I write the ad copy for?\n\n"
            "Try:\n"
            "Product: ...\n"
            "Category: ...\n"
            "Constraints: ... (optional)\n"
            "Task: Full ad / headline only / 5 keywords\n"
        )
    return (
        "Hi! Please describe the product and what you want:\n\n"
        "Product: ...\n"
        "Category: ...\n"
        "Constraints: ... (optional)\n"
        "Task: Full ad / headline only / 5 keywords\n"
    )


def run_agent(user_prompt: str, category_filter: str = "") -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []

    try:
        prompt = (user_prompt or "").strip()
        mode = _detect_mode(prompt)

        guard = {
            "empty": not bool(prompt),
            "too_long": len(prompt) > MAX_PROMPT_CHARS,
            "length": len(prompt),
            "max": MAX_PROMPT_CHARS,
            "mode": mode,
        }
        steps.append({"module": "InputGuard", "prompt": {"user_prompt": user_prompt}, "response": guard})

        if guard["empty"]:
            return {"status": "error", "error": "Empty prompt", "response": None, "steps": steps}
        if guard["too_long"]:
            return {"status": "error", "error": "Prompt too long", "response": None, "steps": steps}

        from_wizard = ("RAG Category Filter:" in prompt) or ("Category:" in prompt) or ("Task:" in prompt)

        if from_wizard:
            clarify, reason = False, ""
        else:
            clarify, reason = _should_clarify(prompt)

        steps.append({
            "module": "IntentGuard",
            "prompt": {"user_prompt": prompt},
            "response": {"should_clarify": clarify, "reason": reason, "from_wizard": from_wizard, "mode": mode},
        })

        if clarify:
            return {"status": "ok", "error": None, "response": _clarification_response(reason), "steps": steps}

        search_query = _rewrite_query(prompt)
        ctx, trace = retrieve_examples(search_query, category_filter=category_filter)

        steps.append({
            "module": "AmazonInspirationRetriever",
            "prompt": {"query": search_query, "original_prompt": prompt, "category_filter": category_filter},
            "response": {
                "provider": trace.provider,
                "index": trace.index_name,
                "namespace": trace.namespace,
                "top_k": trace.top_k,
                "matches": trace.matches,
                "categories": trace.categories,
                "ads_used": trace.ads_used,
                "note": trace.note,
                "ctx_preview": (ctx or "")[:1200],
            },
        })

        condensed = _condense_ctx(prompt, ctx, max_lines=10)
        allowed_terms = _extract_allowed_terms(condensed, max_terms=12)

        system_msg, user_msg = _build_messages(prompt, condensed, allowed_terms, mode=mode)
        llm_text, meta = llm_generate_ad(system_msg, user_msg)

        steps.append({
            "module": "AdCopyWriter",
            "prompt": {"system": system_msg, "user": user_msg},
            "response": meta,
        })

        final_text = (llm_text or "").strip()

        repaired = False
        if ENABLE_REPAIR and not _is_valid_format(final_text, mode):
            fixed, meta2 = _repair_format(prompt, final_text, mode)
            steps.append({
                "module": "FormatRepair",
                "prompt": {"system": "formatter", "user": "(see above)", "mode": mode},
                "response": meta2,
            })
            if _is_valid_format(fixed, mode):
                final_text = fixed.strip()
                repaired = True

        steps.append({
            "module": "FinalResponseComposer",
            "prompt": {"repaired": repaired, "mode": mode},
            "response": {"format_valid": _is_valid_format(final_text, mode)},
        })

        return {"status": "ok", "error": None, "response": final_text, "steps": steps}

    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {str(e)}", "response": None, "steps": steps}