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
        "description": (
            "Ad-Wise is an AI agent that generates high-converting ad copy using RAG over real product listing titles, "
            "and analyzes ad campaign performance metrics against industry benchmarks to provide actionable insights."
        ),
        "purpose": (
            "Help marketers produce (1) a full performance ad listing (headline + 5 bullets + description + keywords + publishing tips), "
            "(2) a headline-only output, (3) exactly 5 must-use keywords for the headline, or "
            "(4) a detailed performance analysis of campaign metrics (CTR, ROI, conversion rate, etc.) compared against "
            "industry benchmarks with actionable recommendations and an improved headline suggestion. "
            "All modes return a full execution trace (steps)."
        ),
        "prompt_template": {
            "template": (
                "AD GENERATION — free-form or structured:\n\n"
                "Product: <describe the product>\n"
                "Category: <optional>\n"
                "RAG Category Filter: <optional>\n"
                "Constraints: <optional>\n"
                "Platform: E-commerce\n"
                "Task: <Full ad / Headline only / 5 keywords>\n\n"
                "PERFORMANCE ANALYSIS — paste metrics naturally:\n\n"
                "Analyze my ad performance: CTR=<value>, ROI=<value>, conversion_rate=<value>, "
                "impressions=<value>, clicks=<value>\n"
                "Or describe it: 'My Facebook campaign had a 3% CTR and 2.5 ROI — how am I doing?'"
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
            },
            {
                "prompt": "Analyze my ad performance: CTR=3%, ROI=2.5, conversion_rate=5%, impressions=50000, clicks=1500",
                "full_response": (
                    "Performance Summary: Your campaign is underperforming across all three key metrics. "
                    "CTR (3.0%) is below the 4.5% benchmark, ROI (2.5) falls short of the 3.2 target, "
                    "and conversion rate (5.0%) lags the 7% benchmark — suggesting both traffic quality and landing page efficiency need attention.\n\n"
                    "Key Issues:\n"
                    "- CTR shortfall: yours=3.0% vs benchmark=4.5% → −1.5 pp (−33% relative); ad creative or targeting is not compelling enough to drive clicks.\n"
                    "- ROI shortfall: yours=2.5 vs benchmark=3.2 → −0.7 (−22% relative); revenue per dollar spent is significantly below target.\n"
                    "- Conversion rate shortfall: yours=5.0% vs benchmark=7.0% → −2.0 pp (−29% relative); visitors are not converting — likely a landing page or offer mismatch.\n\n"
                    "Recommendations:\n"
                    "- Refresh ad creative: A/B test 3 new headlines emphasizing a single strong benefit + urgency CTA; pause bottom 20% of ad sets by CTR.\n"
                    "- Fix conversion funnel: audit landing page load speed, headline-to-ad message match, and CTA placement; run a 2-week A/B test targeting the 7% benchmark.\n"
                    "- Tighten audience targeting: build lookalike audiences from recent converters, apply negative keyword lists, and shift budget to top-performing placements.\n\n"
                    "Suggested Headline: Upgrade Your Results — Shop the #1 Rated Choice and Save Today"
                ),
                "steps": [
                    {"module": "InputGuard", "prompt": {"user_prompt": "Analyze my ad performance: CTR=3%, ROI=2.5, conversion_rate=5%, impressions=50000, clicks=1500"}, "response": {"empty": False, "too_long": False, "length": 98, "max": 4000, "mode": "analyze"}},
                    {"module": "IntentGuard", "prompt": {"user_prompt": "..."}, "response": {"should_clarify": False, "reason": "", "from_wizard": False, "mode": "analyze"}},
                    {"module": "PerformanceParser", "prompt": {"user_prompt": "...", "csv_path": None}, "response": {"source": "text", "error": None, "user_metrics": {"ctr": 3.0, "roi": 2.5, "conversion_rate": 5.0, "impressions": 50000.0, "clicks": 1500.0}, "comparison": {"ctr": {"user_value": 3.0, "benchmark": 4.5, "delta": -1.5, "verdict": "poor"}, "roi": {"user_value": 2.5, "benchmark": 3.2, "delta": -0.7, "verdict": "poor"}, "conversion_rate": {"user_value": 5.0, "benchmark": 7.0, "delta": -2.0, "verdict": "poor"}}}},
                    {"module": "AdCopyWriter", "prompt": {"system": "You are Ad-Wise, an expert digital marketing analyst...", "user": "USER REQUEST: ..."}, "response": {"model": "reasoning", "text_preview": "Performance Summary: Your campaign is underperforming..."}},
                    {"module": "FinalResponseComposer", "prompt": {"repaired": False, "mode": "analyze"}, "response": {"format_valid": True}}
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
