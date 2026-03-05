from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from app.agent import run_agent
from app.conversation_manager import process_message

app = FastAPI(title="Ad-Wise Agent")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
ARCH_PNG   = STATIC_DIR / "architecture.png"
INDEX_HTML = STATIC_DIR / "index.html"


# REQUIRED: execute gets ONLY {"prompt": "..."}
class ExecuteIn(BaseModel):
    prompt: str
    last_prompt: Optional[str] = None  # remembers original product for "continue" flow


# Wizard UI endpoint input
class ChatIn(BaseModel):
    prompt: str
    state: Optional[Dict[str, Any]] = None


# -------------------------
# REQUIRED MAIN ENTRYPOINT
# -------------------------
@app.post("/api/execute")
def execute(inp: ExecuteIn):
    try:
        prompt = (inp.prompt or "").strip()
        if not prompt:
            return {"status": "error", "error": "Empty prompt", "response": None, "steps": []}

        rag_cat = ""
        for line in prompt.splitlines():
            if line.startswith("RAG Category Filter:"):
                rag_cat = line.split(":", 1)[1].strip()
                break

        agent_result = run_agent(prompt, category_filter=rag_cat, last_prompt=inp.last_prompt or "")

        return {
            "status":   agent_result.get("status", "ok"),
            "error":    agent_result.get("error"),
            "response": agent_result.get("response"),
            "steps":    agent_result.get("steps", []),
        }

    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {str(e)}", "response": None, "steps": []}


# -------------------------
# Wizard endpoint
# -------------------------
@app.post("/api/chat")
def chat(inp: ChatIn):
    try:
        new_state, payload = process_message(inp.prompt, inp.state)

        if payload.get("ready"):
            agent_prompt = payload.get("agent_prompt") or ""

            rag_cat = ""
            for line in agent_prompt.splitlines():
                if line.startswith("RAG Category Filter:"):
                    rag_cat = line.split(":", 1)[1].strip()
                    break

            agent_result = run_agent(agent_prompt, category_filter=rag_cat)

            return {
                "status":       agent_result.get("status", "ok"),
                "error":        agent_result.get("error"),
                "response":     agent_result.get("response"),
                "steps":        agent_result.get("steps", []),
                "next_state":   new_state,
                "ui_type":      "result",
                "options":      None,
                "message":      payload.get("message"),
                "agent_prompt": agent_prompt,
            }

        return {
            "status":   "ok",
            "error":    None,
            "response": payload.get("message"),
            "steps": [
                {
                    "module": "ConversationManager",
                    "prompt": {"user_input": inp.prompt, "state": inp.state},
                    "response": {"next_step": new_state.get("step"), "ui_type": payload.get("ui_type")},
                }
            ],
            "next_state": new_state,
            "ui_type":    payload.get("ui_type"),
            "options":    payload.get("options"),
            "message":    payload.get("message"),
        }

    except Exception as e:
        return {
            "status":     "error",
            "error":      f"{type(e).__name__}: {str(e)}",
            "response":   None,
            "steps":      [],
            "next_state": None,
            "ui_type":    "text",
            "options":    None,
            "message":    None,
        }


# -------------------------
# Performance Analysis endpoint
# CSV is parsed client-side in JS and sent as text — no heavy deps needed
# -------------------------
@app.post("/api/analyze")
async def analyze(
    prompt: str = Form(...),
    file: Optional[UploadFile] = File(None),
):
    """
    Analyze ad performance. CSV (if uploaded) is read as raw text and
    prepended to the prompt — all parsing happens in the LLM.
    No pandas/numpy required.
    """
    try:
        prompt_text = (prompt or "").strip()
        if not prompt_text:
            return {"status": "error", "error": "Empty prompt", "response": None, "steps": []}

        # If CSV uploaded, read first 50 rows as raw text and prepend to prompt
        csv_text = ""
        if file and file.filename and file.filename.endswith(".csv"):
            try:
                raw = await file.read()
                lines = raw.decode("utf-8", errors="replace").splitlines()
                csv_text = "\nCSV DATA (first 50 rows):\n" + "\n".join(lines[:51])
            except Exception:
                pass

        full_prompt = f"Analyze my ad performance: {prompt_text}{csv_text}"
        agent_result = run_agent(user_prompt=full_prompt, category_filter="")

        return {
            "status":   agent_result.get("status", "ok"),
            "error":    agent_result.get("error"),
            "response": agent_result.get("response"),
            "steps":    agent_result.get("steps", []),
        }

    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {str(e)}", "response": None, "steps": []}


# -------------------------
# Required info endpoints
# -------------------------
@app.get("/api/team_info")
def team_info():
    return {
        "group_batch_order_number": "2_8",
        "team_name": "Amane_Alaa_Soaad",
        "students": [
            {"name": "Amane Qaddah",  "email": "amane.qaddah@campus.technion.ac.il"},
            {"name": "Alaa Saleh",    "email": "alaa.saleh@campus.technion.ac.il"},
            {"name": "Soaad Hammoud", "email": "soaadhamood@campus.technion.ac.il"},
        ],
    }


@app.get("/api/agent_info")
def agent_info():
    return {
        "description": "Ad-Wise is an AI agent that generates high-converting ad copy using RAG over real product listing titles.",
        "purpose": (
            "Help marketers produce (1) a full performance ad listing, "
            "(2) a headline-only output, or (3) exactly 5 must-use keywords for the headline, "
            "while returning a full execution trace (steps). "
            "Also supports (4) analyzing existing ad performance data against benchmarks."
        ),
        "prompt_template": {
            "template": (
                "You can write naturally (free-form) OR use this structured template:\n\n"
                "Product: <describe the product>\n"
                "Category: <optional>\n"
                "RAG Category Filter: <optional>\n"
                "Constraints: <optional>\n"
                "Platform: E-commerce\n"
                "Task: <Full ad / Headline only / 5 keywords / Analyze performance>\n"
            )
        },
        "prompt_examples": [
            {
                "prompt": (
                    "I'm selling a matte black 1-liter stainless steel insulated water bottle with a leak-proof lid. "
                    "It keeps drinks cold about 24 hours and hot around 12. "
                    "Can you write me a full ad listing (headline + bullets + short description + keywords + publishing tips)?"
                ),
                "full_response": (
                    "Headline: Matte Black 1L Stainless Steel Double-Wall Vacuum Insulated Water Bottle "
                    "— Leak-Proof Lid Keeps Cold ~24 hr & Hot ~12 hr\n"
                    "Bullets:\n"
                    "- 1-liter capacity in matte black stainless steel\n"
                    "- Double-wall vacuum insulated keeps drinks cold about 24 hours and hot around 12\n"
                    "- Leak-proof lid for worry-free carry and storage\n"
                    "- Ideal for sports, travel, hiking and everyday use\n"
                    "- Durable stainless steel construction for long-lasting performance\n"
                    "Short description: Matte black 1L stainless steel insulated water bottle.\n"
                    "Keywords: matte black insulated water bottle, 1 liter stainless steel water bottle\n"
                    "Publishing tips: Use close-up photos of the matte finish and leak-proof lid."
                ),
                "steps": [
                    {"module": "InputGuard", "prompt": {"user_prompt": "..."}, "response": {"empty": False, "too_long": False, "length": 256, "max": 4000, "mode": "full"}},
                    {"module": "IntentGuard", "prompt": {"user_prompt": "..."}, "response": {"should_clarify": False, "reason": "", "from_wizard": False, "mode": "full"}},
                    {"module": "AmazonInspirationRetriever", "prompt": {"query": "...", "original_prompt": "...", "category_filter": ""}, "response": {"provider": "pinecone", "index": "amazon-ads-index", "namespace": "amazon_ads", "top_k": 5, "matches": 5, "categories": ["amazon_sports-outdoors"], "ads_used": 50, "note": "no_filter", "ctx_preview": "..."}},
                    {"module": "AdCopyWriter", "prompt": {"system": "...", "user": "..."}, "response": {"model": "RPRTHPB-gpt-5-mini", "raw_usage": {"completion_tokens": 2231, "prompt_tokens": 714, "total_tokens": 2945}}},
                    {"module": "FinalResponseComposer", "prompt": {"repaired": False, "mode": "full"}, "response": {"format_valid": True}},
                ]
            },
            {
                "prompt": (
                    "Write headline only for a product listing: a wireless ergonomic mouse with silent clicks, "
                    "2.4GHz USB receiver, rechargeable battery, works on Windows and Mac. Headline only please."
                ),
                "full_response": "Headline: Wireless Ergonomic Silent-Click Mouse, 2.4GHz USB Receiver, Rechargeable Battery — Mac & Windows Compatible",
                "steps": [
                    {"module": "InputGuard", "prompt": {"user_prompt": "..."}, "response": {"empty": False, "too_long": False, "length": 180, "max": 4000, "mode": "headline"}},
                    {"module": "IntentGuard", "prompt": {"user_prompt": "..."}, "response": {"should_clarify": False, "reason": "", "from_wizard": False, "mode": "headline"}},
                    {"module": "AmazonInspirationRetriever", "prompt": {"query": "...", "original_prompt": "...", "category_filter": ""}, "response": {"provider": "pinecone", "index": "amazon-ads-index", "namespace": "amazon_ads", "top_k": 5, "matches": 5, "categories": ["amazon_electronics"], "ads_used": 50, "note": "no_filter", "ctx_preview": "..."}},
                    {"module": "AdCopyWriter", "prompt": {"system": "...", "user": "..."}, "response": {"model": "RPRTHPB-gpt-5-mini", "raw_usage": {"completion_tokens": 1124, "prompt_tokens": 638, "total_tokens": 1762}}},
                    {"module": "FinalResponseComposer", "prompt": {"repaired": False, "mode": "headline"}, "response": {"format_valid": True}},
                ]
            },
            {
                "prompt": (
                    "Product: vitamin C face serum with hyaluronic acid — brightening + hydrating, "
                    "fragrance-free, for sensitive skin. "
                    "Please give me 5 keywords only (comma separated) that I must include in the headline."
                ),
                "full_response": "Keywords: Vitamin C, Face Serum, Hyaluronic Acid, Brightening, Sensitive Skin",
                "steps": [
                    {"module": "InputGuard", "prompt": {"user_prompt": "..."}, "response": {"empty": False, "too_long": False, "length": 199, "max": 4000, "mode": "keywords5"}},
                    {"module": "IntentGuard", "prompt": {"user_prompt": "..."}, "response": {"should_clarify": False, "reason": "", "from_wizard": False, "mode": "keywords5"}},
                    {"module": "AmazonInspirationRetriever", "prompt": {"query": "...", "original_prompt": "...", "category_filter": ""}, "response": {"provider": "pinecone", "index": "amazon-ads-index", "namespace": "amazon_ads", "top_k": 5, "matches": 5, "categories": ["amazon_beauty_skin-care"], "ads_used": 50, "note": "no_filter", "ctx_preview": "..."}},
                    {"module": "AdCopyWriter", "prompt": {"system": "...", "user": "..."}, "response": {"model": "RPRTHPB-gpt-5-mini", "raw_usage": {"completion_tokens": 729, "prompt_tokens": 648, "total_tokens": 1377}}},
                    {"module": "FinalResponseComposer", "prompt": {"repaired": False, "mode": "keywords5"}, "response": {"format_valid": True}},
                ]
            },
        ]
    }


@app.get("/api/model_architecture")
def model_architecture():
    if not ARCH_PNG.exists():
        return {"status": "error", "error": "architecture.png not found", "response": None, "steps": []}
    return FileResponse(ARCH_PNG, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def root_ui():
    if not INDEX_HTML.exists():
        return "<h3>Missing static/index.html</h3>"
    return INDEX_HTML.read_text(encoding="utf-8")