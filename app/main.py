from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional
import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup before any requests are accepted.
    Pre-loads the heavy sentence-transformer model so the first
    real request does not time out downloading it from HuggingFace.
    """
    logger.info("==> [Startup] Pre-loading embedding model …")
    try:
        from app.retriever import preload_model
        preload_model()
        logger.info("==> [Startup] Embedding model ready ✓")
    except Exception as e:
        # Don't crash the whole server if model loading fails —
        # the retriever will fall back to FTS or handle the error gracefully.
        logger.warning("==> [Startup] Model pre-load failed (will retry on first request): %s", e)
    yield
    # --- shutdown ---
    logger.info("==> [Shutdown] App stopping.")


app = FastAPI(title="Ad-Wise Agent", lifespan=lifespan)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
ARCH_PNG   = STATIC_DIR / "architecture.png"
INDEX_HTML = STATIC_DIR / "index.html"


# REQUIRED: execute gets ONLY {"prompt": "..."}
class ExecuteIn(BaseModel):
    prompt: str


# Wizard UI endpoint input
class ChatIn(BaseModel):
    prompt: str
    state: Optional[Dict[str, Any]] = None


# -------------------------
# REQUIRED MAIN ENTRYPOINT
# Input:  {"prompt": "..."}
# Output: {"status","error","response","steps"}
# -------------------------
@app.post("/api/execute")
def execute(inp: ExecuteIn):
    try:
        from app.agent import run_agent

        prompt = (inp.prompt or "").strip()
        if not prompt:
            return {"status": "error", "error": "Empty prompt", "response": None, "steps": []}

        rag_cat = ""
        for line in prompt.splitlines():
            if line.startswith("RAG Category Filter:"):
                rag_cat = line.split(":", 1)[1].strip()
                break

        agent_result = run_agent(prompt, category_filter=rag_cat)

        return {
            "status": agent_result.get("status", "ok"),
            "error": agent_result.get("error"),
            "response": agent_result.get("response"),
            "steps": agent_result.get("steps", []),
        }

    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {str(e)}", "response": None, "steps": []}


# -------------------------
# Wizard endpoint for UI buttons/back
# -------------------------
@app.post("/api/chat")
def chat(inp: ChatIn):
    try:
        from app.agent import run_agent
        from app.conversation_manager import process_message

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
                "status": agent_result.get("status", "ok"),
                "error": agent_result.get("error"),
                "response": agent_result.get("response"),
                "steps": agent_result.get("steps", []),
                "next_state": new_state,
                "ui_type": "result",
                "options": None,
                "message": payload.get("message"),
                "agent_prompt": agent_prompt,
            }

        return {
            "status": "ok",
            "error": None,
            "response": payload.get("message"),
            "steps": [
                {
                    "module": "ConversationManager",
                    "prompt": {"user_input": inp.prompt, "state": inp.state},
                    "response": {"next_step": new_state.get("step"), "ui_type": payload.get("ui_type")},
                }
            ],
            "next_state": new_state,
            "ui_type": payload.get("ui_type"),
            "options": payload.get("options"),
            "message": payload.get("message"),
        }

    except Exception as e:
        return {
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
            "response": None,
            "steps": [],
            "next_state": None,
            "ui_type": "text",
            "options": None,
            "message": None,
        }


# Required info endpoints
@app.get("/api/team_info")
def team_info():
    return {
        "group_batch_order_number": "2_8",
        "team_name": "Amane_Alaa_Soaad",
        "students": [
            {"name": "Amane Qaddah",   "email": "amane.qaddah@campus.technion.ac.il"},
            {"name": "Alaa Saleh",     "email": "alaa.saleh@campus.technion.ac.il"},
            {"name": "Soaad Hammoud",  "email": "soaadhamood@campus.technion.ac.il"},
        ],
    }


@app.get("/api/agent_info")
def agent_info():
    return {
        "description": "Ad-Wise is an AI agent that generates high-converting ad copy using RAG over real product listing titles.",
        "purpose": (
            "Help marketers produce (1) a full performance ad listing, "
            "(2) a headline-only output, or (3) exactly 5 must-use keywords for the headline, "
            "while returning a full execution trace (steps)."
        ),
        "prompt_template": {
            "template": (
                "You can write naturally (free-form) OR use this structured template:\n\n"
                "Product: <describe the product>\n"
                "Category: <optional>\n"
                "RAG Category Filter: <optional>\n"
                "Constraints: <optional>\n"
                "Platform: E-commerce\n"
                "Task: <Full ad / Headline only / 5 keywords>\n"
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
                    "Headline: Matte Black 1L Stainless Steel Insulated Water Bottle – 24H Cold, 12H Hot, Leak-Proof Lid\n"
                    "Bullets:\n"
                    "- STAY COLD 24 HOURS: Double-wall vacuum insulation keeps your drinks ice-cold all day long\n"
                    "- KEEP HOT 12 HOURS: Perfect for coffee, tea, or hot cocoa on the go\n"
                    "- LEAK-PROOF LID: Secure seal means zero spills in your bag or car\n"
                    "- 1-LITER CAPACITY: Hydrate all day without constant refills\n"
                    "- MATTE BLACK FINISH: Sleek, scratch-resistant coating that looks great anywhere\n"
                    "Short description: The ultimate everyday bottle — 1L stainless steel insulation keeps drinks cold 24H or hot 12H with a 100% leak-proof lid.\n"
                    "Keywords: insulated water bottle, stainless steel water bottle, leak proof water bottle, 1 liter water bottle, matte black water bottle, vacuum insulated bottle, cold 24 hours hot 12 hours, double wall bottle\n"
                    "Publishing tips: Use 'cold 24H hot 12H' in the title for search; A+ content should show the bottle in outdoor/gym settings; run Sponsored Products on 'insulated water bottle' and 'leak proof bottle'."
                ),
                "steps": [
                    {"module": "InputGuard", "prompt": {"user_prompt": "..."}, "response": {"empty": False, "too_long": False, "length": 210, "max": 4000, "mode": "full"}},
                    {"module": "IntentGuard", "prompt": {"user_prompt": "..."}, "response": {"should_clarify": False, "reason": "", "from_wizard": False, "mode": "full"}},
                    {"module": "AmazonInspirationRetriever", "prompt": {"query": "stainless steel insulated water bottle leak proof", "category_filter": ""}, "response": {"provider": "pinecone", "matches": 5, "ads_used": 40, "note": "no_filter"}},
                    {"module": "AdCopyWriter", "prompt": {"system": "You are Ad-Wise...", "user": "USER REQUEST: ..."}, "response": {"model": "reasoning", "text_preview": "Headline: Matte Black..."}},
                    {"module": "FinalResponseComposer", "prompt": {"repaired": False, "mode": "full"}, "response": {"format_valid": True}}
                ]
            }
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