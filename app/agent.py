from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from app.retriever import retrieve_examples
from app.llm_client import llm_generate_ad
from app.settings import ENABLE_REPAIR, MAX_PROMPT_CHARS


_MODE_FULL      = "full"
_MODE_HEADLINE  = "headline"
_MODE_KEYWORDS5 = "keywords5"
_MODE_ANALYZE   = "analyze"   # ← NEW


# ------------------------------------------------------------------ #
# Mode detection
# ------------------------------------------------------------------ #

_ANALYZE_HINTS = {
    "analyze", "analyse", "performance", "ctr", "roi", "conversion rate",
    "engagement rate", "impressions", "acquisition cost",
    "how is my ad", "how did my campaign", "review my ad",
    "my campaign", "my ad performance", "benchmark", "compare my",
}

def _detect_mode(prompt: str) -> str:
    p = (prompt or "").lower()

    # Analyze mode – check before other modes
    if any(h in p for h in _ANALYZE_HINTS):
        return _MODE_ANALYZE

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


# ------------------------------------------------------------------ #
# Format validation (existing modes)
# ------------------------------------------------------------------ #

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
    if mode == _MODE_ANALYZE:
        return len(t) > 50
    return bool(_FULL_FORMAT_REGEX.fullmatch(t))


# ------------------------------------------------------------------ #
# Shared helpers (existing)
# ------------------------------------------------------------------ #

_STOPWORDS = {
    "a","an","the","and","or","for","to","of","in","on","with","without","by",
    "include","includes","including","style","amazon","ad","write","bullet","bullets",
    "cta","benefits","your","now","next","new","please","make","create",
    "this","that","these","those","from","into","over","under","best","top",
    "high","quality","premium","great"
}

_INTENT_HINTS  = {"write", "generate", "create", "headline", "keywords", "ad", "listing", "product"}
_PRODUCT_HINTS = {
    "bottle","shoe","shoes","watch","backpack","headphones","laptop","charger","camera",
    "mug","cup","speaker","mouse","keyboard","skincare","perfume","dress","book","books",
    "shirt","pants","jacket","bag","case","cover","stand","mat","pad","ring","bracelet",
    "necklace","earring","pillow","blanket","towel","lamp","chair","desk","pen","notebook",
    "tablet","phone","cable","adapter","battery","brush","comb","razor","serum",
    "cream","lotion","shampoo","soap","candle","toy","game","ball","bike","tent",
    "knife","pot","pan","tray","rack","shelf","mirror","clock","wallet","belt","hat","cap",
    "gloves","socks","scarf","umbrella","sunglasses","helmet","lock","drill","vacuum","blender",
}

MIN_PROMPT_CHARS_FOR_AGENT = 4
MIN_TOKENS_FOR_AGENT       = 1


def _rewrite_query(prompt: str) -> str:
    category_part = ""
    product_part  = ""
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

    seen   = set()
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
    toks   = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", condensed_ctx.lower())
    toks   = [t for t in toks if t not in _STOPWORDS]
    common = [w for w, c in Counter(toks).most_common(80) if c >= 2]
    return common[:max_terms]


def _build_messages(prompt: str, condensed_ctx: str, allowed_terms: List[str], mode: str) -> Tuple[str, str]:
    system_msg = (
        "You are Ad-Wise, an expert performance copywriter specializing in Amazon and e-commerce ads. "
        "Write high-converting ad copy using the provided inspiration examples. "
        "Be specific, vivid, and benefit-focused — never repeat the same idea across bullets. "
        "Each bullet must highlight a DIFFERENT benefit or feature. "
        "Even with minimal product info, infer realistic benefits from the product type and category. "
        "Avoid generic filler like 'a great product' or restating the product name five times. "
        "Follow user constraints strictly."
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
            "- ...\n- ...\n- ...\n- ...\n- ...\n"
            "Short description: ...\n"
            "Keywords: ...\n"
            "Publishing tips: ..."
        )

    user_msg = (
        f"Required format:\n{required}\n\n"
        f"User request:\n{product_prompt}\n\n"
        f"Draft to fix:\n{draft}"
    )

    return llm_generate_ad(system_msg, user_msg)


_CONTINUE_PHRASES = {"continue","just continue","go ahead","generate anyway","proceed","do it anyway","yes continue","yes","ok","sure"}

def _is_continue_signal(prompt: str) -> bool:
    return (prompt or "").strip().lower() in _CONTINUE_PHRASES


def _should_clarify(prompt: str) -> Tuple[bool, str]:
    p = (prompt or "").strip()
    if not p:
        return True, "empty"
    if _is_continue_signal(p):
        return False, ""
    if len(p) < MIN_PROMPT_CHARS_FOR_AGENT:
        return True, "too_short"

    p_l = p.lower()
    toks = re.findall(r"[a-z0-9]+", p_l)
    meaningful = [t for t in toks if len(t) >= 2]
    has_intent  = any(h in p_l for h in _INTENT_HINTS)
    has_product = any(h in p_l for h in _PRODUCT_HINTS) or ("product:" in p_l)

    # Rich description — generate immediately
    if len(meaningful) >= 8:
        return False, ""
    has_descriptor = bool(re.search(
        r"(color|colour|size|material|made|designed|features?|includes?|with|for|about|"
        r"ml|oz|liter|litre|inch|cm|mm|kg|lb|red|blue|green|black|white|yellow|pink|"
        r"wooden|plastic|metal|steel|leather|cotton|organic|waterproof|wireless|portable)",
        p_l
    ))
    if has_descriptor and len(meaningful) >= 4:
        return False, ""

    if has_intent and not has_product:
        return True, "missing_product"
    return True, "needs_details"


def _clarification_response(reason: str = "", original_prompt: str = "") -> str:
    product_hint = f'"{original_prompt}"' if original_prompt else "your input"
    if reason == "missing_product":
        return (
            f"I'd love to help! Could you tell me more about the product?\n\n"
            f"Try adding:\n"
            f"• Product name / model\n"
            f"• Key features (material, size, color, specs)\n"
            f"• Target audience or unique selling point\n\n"
            f"Or type **continue** and I'll write an ad based on {product_hint} as-is."
        )
    return (
        f"Got it — **{original_prompt}**! To write the best ad, could you share a few more details?\n\n"
        f"For example:\n"
        f"• Key features (material, size, color, capacity)\n"
        f"• What makes it unique or better than competitors?\n"
        f"• Who is the target audience?\n\n"
        f"Or type **continue** to generate an ad based on {product_hint} right away."
    )


# ------------------------------------------------------------------ #
# NEW: Analyze mode pipeline
# ------------------------------------------------------------------ #

def _build_analyze_messages(user_prompt: str, analysis_context: str) -> Tuple[str, str]:
    system_msg = (
        "You are Ad-Wise, an expert digital marketing analyst and copywriter. "
        "You analyze ad campaign performance data, compare it against benchmarks, "
        "and provide actionable insights plus improved ad copy suggestions. "
        "Be specific, data-driven, and concise."
    )

    benchmarks = (
        "Industry benchmarks:\n"
        "- CTR: 4.5% (good) / below 2% (poor)\n"
        "- ROI: 3.2 (good) / below 1.5 (poor)\n"
        "- Conversion Rate: 7% (good) / below 3% (poor)\n"
        "- Acquisition Cost: under $15 (good) / above $25 (poor)\n"
    )
    context_block = f"{analysis_context}\n\n{benchmarks}" if analysis_context else benchmarks

    user_msg = (
        f"USER REQUEST:\n{user_prompt}\n\n"
        f"{context_block}\n\n"
        "Extract any metrics mentioned, compare against the benchmarks above, and provide:\n\n"
        "Performance Summary: <2-3 sentences on overall performance vs benchmark>\n"
        "Key Issues:\n"
        "- <specific issue with metric and delta>\n"
        "- <specific issue with metric and delta>\n"
        "- <specific issue with metric and delta>\n"
        "Recommendations:\n"
        "- <actionable recommendation 1>\n"
        "- <actionable recommendation 2>\n"
        "- <actionable recommendation 3>\n"
        "Suggested Headline: <one improved ad headline based on the data>\n"
    )

    return system_msg, user_msg


def _run_analyze_pipeline(
    prompt: str,
    steps: List[Dict[str, Any]],
    csv_path: Optional[str] = None,
) -> Dict[str, Any]:

    try:
        from app.analyzer import analyze_performance
    except ImportError:
        return {
            "status": "error",
            "error": "analyzer module not found. Make sure app/analyzer.py exists.",
            "response": None,
            "steps": steps,
        }

    # Step: PerformanceParser
    analysis = analyze_performance(user_input=prompt, csv_path=csv_path)

    steps.append({
        "module": "PerformanceParser",
        "prompt": {"user_prompt": prompt, "csv_path": csv_path},
        "response": {
            "source":       analysis.get("source", "text"),
            "error":        analysis.get("error"),
            "user_metrics": analysis.get("user_metrics", {}),
            "comparison":   analysis.get("comparison", {}),
        },
    })

    if analysis.get("error") and not analysis.get("user_metrics"):
        return {
            "status": "error",
            "error": analysis["error"],
            "response": None,
            "steps": steps,
        }

    # Step: AdCopyWriter (analyze flavor)
    system_msg, user_msg = _build_analyze_messages(prompt, analysis.get("context_for_llm", ""))
    try:
        llm_text, meta = llm_generate_ad(system_msg, user_msg)
    except Exception as llm_err:
        return {
            "status": "error",
            "error": f"LLM call failed: {type(llm_err).__name__}: {llm_err}",
            "response": None,
            "steps": steps,
        }

    steps.append({
        "module": "AdCopyWriter",
        "prompt": {"system": system_msg, "user": user_msg},
        "response": meta,
    })

    final_text = (llm_text or "").strip()

    if not final_text:
        return {
            "status": "error",
            "error": "LLM returned an empty response. Please try again with more detailed metrics.",
            "response": None,
            "steps": steps,
        }

    steps.append({
        "module": "FinalResponseComposer",
        "prompt": {"repaired": False, "mode": _MODE_ANALYZE},
        "response": {"format_valid": _is_valid_format(final_text, _MODE_ANALYZE)},
    })

    return {"status": "ok", "error": None, "response": final_text, "steps": steps}


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def run_agent(
    user_prompt: str,
    category_filter: str = "",
    csv_path: Optional[str] = None,
    last_prompt: str = "",
) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []

    try:
        prompt = (user_prompt or "").strip()
        mode   = _detect_mode(prompt)

        guard = {
            "empty":    not bool(prompt),
            "too_long": len(prompt) > MAX_PROMPT_CHARS,
            "length":   len(prompt),
            "max":      MAX_PROMPT_CHARS,
            "mode":     mode,
        }
        steps.append({"module": "InputGuard", "prompt": {"user_prompt": user_prompt}, "response": guard})

        if guard["empty"]:
            return {"status": "error", "error": "Empty prompt", "response": None, "steps": steps}
        if guard["too_long"]:
            return {"status": "error", "error": "Prompt too long", "response": None, "steps": steps}

        # ── Analyze mode ───────────────────────────────────────────
        if mode == _MODE_ANALYZE:
            steps.append({
                "module": "IntentGuard",
                "prompt": {"user_prompt": prompt},
                "response": {
                    "should_clarify": False,
                    "reason": "",
                    "from_wizard": False,
                    "mode": mode,
                },
            })
            # Try the full analyzer pipeline first; fall back to direct LLM if unavailable
            try:
                from app.analyzer import analyze_performance
                return _run_analyze_pipeline(prompt, steps, csv_path=csv_path)
            except ImportError:
                pass
            # Fallback: send directly to LLM with an analysis-focused system prompt
            system_msg, user_msg = _build_analyze_messages(prompt, "")
            try:
                llm_text, meta = llm_generate_ad(system_msg, user_msg)
            except Exception as llm_err:
                return {"status": "error", "error": f"LLM call failed: {type(llm_err).__name__}: {llm_err}", "response": None, "steps": steps}
            steps.append({"module": "AdCopyWriter", "prompt": {"system": system_msg, "user": user_msg}, "response": meta})
            final_text = (llm_text or "").strip()
            if not final_text:
                return {"status": "error", "error": "LLM returned an empty response. Please try again.", "response": None, "steps": steps}
            steps.append({"module": "FinalResponseComposer", "prompt": {"mode": _MODE_ANALYZE}, "response": {"format_valid": bool(final_text)}})
            return {"status": "ok", "error": None, "response": final_text, "steps": steps}

        # ── Ad generation modes (existing logic) ───────────────────
        from_wizard = ("RAG Category Filter:" in prompt) or ("Category:" in prompt) or ("Task:" in prompt)
        is_detail_turn = bool(last_prompt) and not _is_continue_signal(prompt) and not from_wizard

        if from_wizard or is_detail_turn:
            clarify, reason = False, ""
        else:
            clarify, reason = _should_clarify(prompt)

        steps.append({
            "module": "IntentGuard",
            "prompt": {"user_prompt": prompt},
            "response": {"should_clarify": clarify, "reason": reason, "from_wizard": from_wizard, "mode": mode},
        })

        if clarify:
            return {"status": "ok", "error": None, "response": _clarification_response(reason, prompt), "steps": steps}

        # "continue" → use original product word
        if _is_continue_signal(prompt):
            original = (last_prompt or "").strip()
            if original and not _is_continue_signal(original):
                prompt = f"Product: {original} — write a full high-converting ad"
            else:
                prompt = "Product: (unspecified product) — write a general high-converting ad"
            mode = _MODE_FULL
        # detail turn → merge original + new details
        elif is_detail_turn:
            original = (last_prompt or "").strip()
            if original and len(original.split()) <= 6:
                prompt = f"Product: {original}. Additional details: {prompt}"

        search_query = _rewrite_query(prompt)
        ctx, trace   = retrieve_examples(search_query, category_filter=category_filter)

        steps.append({
            "module": "AmazonInspirationRetriever",
            "prompt": {"query": search_query, "original_prompt": prompt, "category_filter": category_filter},
            "response": {
                "provider":    trace.provider,
                "index":       trace.index_name,
                "namespace":   trace.namespace,
                "top_k":       trace.top_k,
                "matches":     trace.matches,
                "categories":  trace.categories,
                "ads_used":    trace.ads_used,
                "note":        trace.note,
                "ctx_preview": (ctx or "")[:1200],
            },
        })

        condensed     = _condense_ctx(prompt, ctx, max_lines=10)
        allowed_terms = _extract_allowed_terms(condensed, max_terms=12)

        system_msg, user_msg = _build_messages(prompt, condensed, allowed_terms, mode=mode)
        llm_text, meta       = llm_generate_ad(system_msg, user_msg)

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
                repaired   = True

        steps.append({
            "module": "FinalResponseComposer",
            "prompt": {"repaired": repaired, "mode": mode},
            "response": {"format_valid": _is_valid_format(final_text, mode)},
        })

        return {"status": "ok", "error": None, "response": final_text, "steps": steps}

    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {str(e)}", "response": None, "steps": steps}
