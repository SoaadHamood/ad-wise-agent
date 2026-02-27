from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
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
        prompt = (inp.prompt or "").strip()
        if not prompt:
            return {"status": "error", "error": "Empty prompt", "response": None, "steps": []}

        rag_cat = ""
        for line in prompt.splitlines():
            if line.startswith("RAG Category Filter:"):
                rag_cat = line.split(":", 1)[1].strip()
                break

        agent_result = run_agent(prompt, category_filter=rag_cat)

        # MUST match exactly these top-level fields:
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
            # -------------------------
            # Example 1: FULL AD
            # -------------------------
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
                    "Short description: Matte black 1L stainless steel insulated water bottle with double-wall vacuum insulation "
                    "and a leak-proof lid — keeps drinks cold ~24 hours and hot ~12 hours for sports and travel.\n"
                    "Keywords: matte black insulated water bottle, 1 liter stainless steel water bottle, double wall vacuum bottle, "
                    "leak-proof lid water bottle, keeps drinks cold 24 hours, keeps drinks hot 12 hours, sports insulated water bottle, "
                    "travel stainless steel bottle, outdoor vacuum insulated bottle, black stainless bottle\n"
                    "Publishing tips: Use close-up photos of the matte finish and leak-proof lid; highlight cold/hot duration "
                    "in the first bullet and product images; include lifestyle shots showing sports and travel use."
                ),
                "steps": [
                    {
                        "module": "InputGuard",
                        "prompt": {
                            "user_prompt": (
                                "I'm selling a matte black 1-liter stainless steel insulated water bottle with a leak-proof lid. "
                                "It keeps drinks cold about 24 hours and hot around 12. "
                                "Can you write me a full ad listing (headline + bullets + short description + keywords + publishing tips)?"
                            )
                        },
                        "response": {"empty": False, "too_long": False, "length": 256, "max": 4000, "mode": "full"}
                    },
                    {
                        "module": "IntentGuard",
                        "prompt": {
                            "user_prompt": (
                                "I'm selling a matte black 1-liter stainless steel insulated water bottle with a leak-proof lid. "
                                "It keeps drinks cold about 24 hours and hot around 12. "
                                "Can you write me a full ad listing (headline + bullets + short description + keywords + publishing tips)?"
                            )
                        },
                        "response": {"should_clarify": False, "reason": "", "from_wizard": False, "mode": "full"}
                    },
                    {
                        "module": "AmazonInspirationRetriever",
                        "prompt": {
                            "query": "selling matte black liter stainless steel insulated water bottle leak proof lid",
                            "original_prompt": (
                                "I'm selling a matte black 1-liter stainless steel insulated water bottle with a leak-proof lid. "
                                "It keeps drinks cold about 24 hours and hot around 12. "
                                "Can you write me a full ad listing (headline + bullets + short description + keywords + publishing tips)?"
                            ),
                            "category_filter": ""
                        },
                        "response": {
                            "provider": "pinecone", "index": "amazon-ads-index", "namespace": "amazon_ads",
                            "top_k": 5, "matches": 5,
                            "categories": ["amazon_aports-outdoors_recreation_camping-hiking_hydration-filtration"],
                            "ads_used": 50, "note": "no_filter",
                            "ctx_preview": (
                                "[Category: amazon_aports-outdoors_recreation_camping-hiking_hydration-filtration]\n"
                                "- Chelii Water Bottle Double Wall Vacuum Insulated Stainless Steel Bottle Cup Keeps Hot and Cold Drinks\n"
                                "- MAKERSLAND Stainless Steel Vacuum Insulated Sport Water Bottle | Leak-Proof Double Walled Cola Shape Bottle\n"
                                "- Mizu - V8 Water Bottle | 26 oz. Double Wall Stainless Steel Vacuum Insulated | Narrow Mouth with Leak Proof Cap\n"
                                "- Glaciar Protein Shaker Bottle with 2 compartments Stainless Steel Holds Drinks hot and Cold Leak Proof\n"
                                "- Aquatix (Black 21 Ounce) Pure Stainless Steel Double Wall Vacuum Insulated Sports Water Bottle"
                            )
                        }
                    },
                    {
                        "module": "AdCopyWriter",
                        "prompt": {
                            "system": (
                                "You are Ad-Wise, an expert performance copywriter. "
                                "Write high-converting ad copy using the provided inspiration examples. "
                                "Be concise, avoid fluff, and follow user constraints strictly."
                            ),
                            "user": (
                                "USER REQUEST:\n"
                                "I'm selling a matte black 1-liter stainless steel insulated water bottle with a leak-proof lid. "
                                "It keeps drinks cold about 24 hours and hot around 12. "
                                "Can you write me a full ad listing (headline + bullets + short description + keywords + publishing tips)?\n\n"
                                "ALLOWED CLAIM TERMS: bottle, stainless, steel, water, double, wall, vacuum, insulated, hot, cold, sports, keeps\n\n"
                                "OUTPUT FORMAT: Headline / Bullets (5) / Short description / Keywords / Publishing tips"
                            )
                        },
                        "response": {
                            "used_url": "https://api.llmod.ai/v1/chat/completions",
                            "model": "RPRTHPB-gpt-5-mini",
                            "text_preview": (
                                "Headline: Matte Black 1L Stainless Steel Double-Wall Vacuum Insulated Water Bottle "
                                "— Leak-Proof Lid Keeps Cold ~24 hr & Hot ~12 hr"
                            ),
                            "raw_usage": {"completion_tokens": 2231, "prompt_tokens": 714, "total_tokens": 2945}
                        }
                    },
                    {
                        "module": "FinalResponseComposer",
                        "prompt": {"repaired": False, "mode": "full"},
                        "response": {"format_valid": True}
                    }
                ]
            },

            # -------------------------
            # Example 2: HEADLINE ONLY
            # -------------------------
            {
                "prompt": (
                    "Write headline only for a product listing: a wireless ergonomic mouse with silent clicks, "
                    "2.4GHz USB receiver, rechargeable battery, works on Windows and Mac. Headline only please."
                ),
                "full_response": (
                    "Headline: Wireless Ergonomic Silent-Click Mouse, 2.4GHz USB Receiver, "
                    "Rechargeable Battery — Mac & Windows Compatible"
                ),
                "steps": [
                    {
                        "module": "InputGuard",
                        "prompt": {
                            "user_prompt": (
                                "Write headline only for a product listing: a wireless ergonomic mouse with silent clicks, "
                                "2.4GHz USB receiver, rechargeable battery, works on Windows and Mac. Headline only please."
                            )
                        },
                        "response": {"empty": False, "too_long": False, "length": 180, "max": 4000, "mode": "headline"}
                    },
                    {
                        "module": "IntentGuard",
                        "prompt": {
                            "user_prompt": (
                                "Write headline only for a product listing: a wireless ergonomic mouse with silent clicks, "
                                "2.4GHz USB receiver, rechargeable battery, works on Windows and Mac. Headline only please."
                            )
                        },
                        "response": {"should_clarify": False, "reason": "", "from_wizard": False, "mode": "headline"}
                    },
                    {
                        "module": "AmazonInspirationRetriever",
                        "prompt": {
                            "query": "headline only product listing wireless ergonomic mouse silent clicks 4ghz usb receiver",
                            "original_prompt": (
                                "Write headline only for a product listing: a wireless ergonomic mouse with silent clicks, "
                                "2.4GHz USB receiver, rechargeable battery, works on Windows and Mac. Headline only please."
                            ),
                            "category_filter": ""
                        },
                        "response": {
                            "provider": "pinecone", "index": "amazon-ads-index", "namespace": "amazon_ads",
                            "top_k": 5, "matches": 5,
                            "categories": [
                                "amazon_aports-outdoors_recreation_camping-hiking_navigation-electronics",
                                "amazon_electronics_headphones"
                            ],
                            "ads_used": 50, "note": "no_filter",
                            "ctx_preview": (
                                "[Category: amazon_aports-outdoors_recreation_camping-hiking_navigation-electronics]\n"
                                "- Emopeak Silent Wireless Mouse E2Pro Noiseless Click with 2.4G Optical Mice 3 Adjustable DPI Levels with USB Receiver\n"
                                "- EDUP LOVE USB 3.0 WiFi Adapter AC1300Mbps Dual Band 5GHz 2.4GHz for Mac OS Windows\n"
                                "- Logitech MK710-RB Desktop Wireless Keyboard/Mouse Combo Wireless Mouse USB Black (Renewed)\n"
                                "- Number Pad Portable Mini USB 2.4GHz 19-Key Numeric Keypad for Laptop PC Desktop"
                            )
                        }
                    },
                    {
                        "module": "AdCopyWriter",
                        "prompt": {
                            "system": (
                                "You are Ad-Wise, an expert performance copywriter. "
                                "Write high-converting ad copy using the provided inspiration examples. "
                                "Be concise, avoid fluff, and follow user constraints strictly."
                            ),
                            "user": (
                                "USER REQUEST:\n"
                                "Write headline only for a product listing: a wireless ergonomic mouse with silent clicks, "
                                "2.4GHz USB receiver, rechargeable battery, works on Windows and Mac. Headline only please.\n\n"
                                "ALLOWED CLAIM TERMS: usb, wireless, ghz, mac, adapter, desktop, keyboard, key, mouse, wifi, dual, windows\n\n"
                                "OUTPUT FORMAT: Headline: ..."
                            )
                        },
                        "response": {
                            "used_url": "https://api.llmod.ai/v1/chat/completions",
                            "model": "RPRTHPB-gpt-5-mini",
                            "text_preview": (
                                "Headline: Wireless Ergonomic Silent-Click Mouse, 2.4GHz USB Receiver, "
                                "Rechargeable Battery — Mac & Windows Compatible"
                            ),
                            "raw_usage": {"completion_tokens": 1124, "prompt_tokens": 638, "total_tokens": 1762}
                        }
                    },
                    {
                        "module": "FinalResponseComposer",
                        "prompt": {"repaired": False, "mode": "headline"},
                        "response": {"format_valid": True}
                    }
                ]
            },

            # -------------------------
            # Example 3: 5 KEYWORDS ONLY
            # -------------------------
            {
                "prompt": (
                    "Product: vitamin C face serum with hyaluronic acid — brightening + hydrating, "
                    "fragrance-free, for sensitive skin. "
                    "Please give me 5 keywords only (comma separated) that I must include in the headline."
                ),
                "full_response": "Keywords: Vitamin C, Face Serum, Hyaluronic Acid, Brightening, Sensitive Skin",
                "steps": [
                    {
                        "module": "InputGuard",
                        "prompt": {
                            "user_prompt": (
                                "Product: vitamin C face serum with hyaluronic acid — brightening + hydrating, "
                                "fragrance-free, for sensitive skin. "
                                "Please give me 5 keywords only (comma separated) that I must include in the headline."
                            )
                        },
                        "response": {"empty": False, "too_long": False, "length": 199, "max": 4000, "mode": "keywords5"}
                    },
                    {
                        "module": "IntentGuard",
                        "prompt": {
                            "user_prompt": (
                                "Product: vitamin C face serum with hyaluronic acid — brightening + hydrating, "
                                "fragrance-free, for sensitive skin. "
                                "Please give me 5 keywords only (comma separated) that I must include in the headline."
                            )
                        },
                        "response": {"should_clarify": False, "reason": "", "from_wizard": False, "mode": "keywords5"}
                    },
                    {
                        "module": "AmazonInspirationRetriever",
                        "prompt": {
                            "query": "product vitamin face serum hyaluronic acid brightening hydrating fragrance free sensitive skin",
                            "original_prompt": (
                                "Product: vitamin C face serum with hyaluronic acid — brightening + hydrating, "
                                "fragrance-free, for sensitive skin. "
                                "Please give me 5 keywords only (comma separated) that I must include in the headline."
                            ),
                            "category_filter": ""
                        },
                        "response": {
                            "provider": "pinecone", "index": "amazon-ads-index", "namespace": "amazon_ads",
                            "top_k": 5, "matches": 5,
                            "categories": ["amazon_beauty_skin-care"],
                            "ads_used": 50, "note": "no_filter",
                            "ctx_preview": (
                                "[Category: amazon_beauty_skin-care]\n"
                                "- BioTrust Ageless Glow Anti Aging Moisturizer Skin Brightening Serum with Vitamin C and Hyaluronic Acid\n"
                                "- Neutrogena Hydro Boost Gentle Cleansing and Hydrating Face Lotion Oil-Free for Sensitive Skin\n"
                                "- Face Mask Gel by Olay Masks Overnight Facial Moisturizer with Vitamin A and Hyaluronic Acid\n"
                                "- Cosmedica Skincare Best-Seller Set- Vitamin C Super Serum and Pure Hyaluronic Acid\n"
                                "- Hyaluronic Acid Serum with Vitamin C A D E ~ Best Anti Aging Cream & Anti Wrinkle Moisturizer"
                            )
                        }
                    },
                    {
                        "module": "AdCopyWriter",
                        "prompt": {
                            "system": (
                                "You are Ad-Wise, an expert performance copywriter. "
                                "Write high-converting ad copy using the provided inspiration examples. "
                                "Be concise, avoid fluff, and follow user constraints strictly."
                            ),
                            "user": (
                                "USER REQUEST:\n"
                                "Product: vitamin C face serum with hyaluronic acid — brightening + hydrating, "
                                "fragrance-free, for sensitive skin. "
                                "Please give me 5 keywords only (comma separated) that I must include in the headline.\n\n"
                                "ALLOWED CLAIM TERMS: skin, anti, face, vitamin, acid, serum, hyaluronic, aging, moisturizer, mask, brightening, facial\n\n"
                                "OUTPUT FORMAT: Keywords: k1, k2, k3, k4, k5"
                            )
                        },
                        "response": {
                            "used_url": "https://api.llmod.ai/v1/chat/completions",
                            "model": "RPRTHPB-gpt-5-mini",
                            "text_preview": "Keywords: Vitamin C, Face Serum, Hyaluronic Acid, Brightening, Sensitive Skin",
                            "raw_usage": {"completion_tokens": 729, "prompt_tokens": 648, "total_tokens": 1377}
                        }
                    },
                    {
                        "module": "FinalResponseComposer",
                        "prompt": {"repaired": False, "mode": "keywords5"},
                        "response": {"format_valid": True}
                    }
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