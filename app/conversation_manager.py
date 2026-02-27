from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

MENU_ACTIONS: List[Dict[str, str]] = [
    {"id": "full_ad",       "label": "✍️  Create Full Ad",        "desc": "Headline + 5 bullets + description + keywords + publishing tips"},
    {"id": "headline_only", "label": "🏷️  Create Headline Only",  "desc": "One high-converting headline line"},
    {"id": "keywords_5",    "label": "🔑  Get 5 Must-Use Keywords","desc": "Exactly 5 keywords/phrases to include in the headline"},
]

CATEGORY_TREE: List[Dict] = [
    {"id": "electronics", "label": "💻  Electronics & Computers", "pinecone_id": "electronics",
     "subcategories": [{"id":"electronics","label":"Electronics (General)","pinecone_id":"electronics"},
                       {"id":"computers","label":"Computers","pinecone_id":"computers"}]},
    {"id": "home_kitchen", "label": "🏠  Home & Kitchen", "pinecone_id": "home-kitchen", "subcategories": []},
    {"id": "sports_outdoors", "label": "⚽  Sports & Outdoors", "pinecone_id": "sports-outdoors", "subcategories": []},
    {"id": "beauty_health", "label": "💄  Beauty & Health", "pinecone_id": "health-household",
     "subcategories": [{"id":"beauty","label":"Beauty","pinecone_id":"beauty"},
                       {"id":"health-household","label":"Health & Household","pinecone_id":"health-household"}]},
    {"id": "automotive", "label": "🚗  Automotive", "pinecone_id": "automotive", "subcategories": []},
    {"id": "baby", "label": "🍼  Baby", "pinecone_id": "baby", "subcategories": []},
    {"id": "pets", "label": "🐾  Pets", "pinecone_id": "pets", "subcategories": []},
    {"id": "luggage", "label": "🧳  Luggage & Travel", "pinecone_id": "luggage", "subcategories": []},
    {"id": "arts_crafts", "label": "🎨  Arts & Crafts", "pinecone_id": "arts_crafts", "subcategories": []},
    {"id": "industrial", "label": "🏭  Industrial & Scientific", "pinecone_id": "industrial-scientific", "subcategories": []},
]

BACK_COMMANDS = {"__back", "back", "go back", "prev", "previous", "return", "חזור", "אחורה"}


def _empty_state() -> Dict[str, Any]:
    return {
        "step":        "GREETING",
        "action":      None,
        "category":    None,
        "subcategory": None,
        "product":     None,
        "constraints": None,
    }


def _top_level_options() -> List[Dict[str, str]]:
    return [{"id": c["id"], "label": c["label"]} for c in CATEGORY_TREE]


def _subcategory_options(cat_id: str) -> List[Dict[str, str]]:
    for c in CATEGORY_TREE:
        if c["id"] == cat_id:
            return [{"id": s["id"], "label": s["label"]} for s in c.get("subcategories", [])]
    return []


def _has_subcategories(cat_id: str) -> bool:
    for c in CATEGORY_TREE:
        if c["id"] == cat_id:
            return len(c.get("subcategories", [])) > 1
    return False


def _pinecone_id(cat_id: str, sub_id: Optional[str]) -> str:
    if sub_id:
        for c in CATEGORY_TREE:
            for s in c.get("subcategories", []):
                if s["id"] == sub_id:
                    return s["pinecone_id"]
    for c in CATEGORY_TREE:
        if c["id"] == cat_id:
            return c["pinecone_id"]
    return cat_id


def _category_label(cat_id: str, sub_id: Optional[str]) -> str:
    for c in CATEGORY_TREE:
        if c["id"] == cat_id:
            if sub_id:
                for s in c.get("subcategories", []):
                    if s["id"] == sub_id:
                        return f"{c['label'].split('  ')[1]} > {s['label']}"
            return c["label"].split("  ")[1] if "  " in c["label"] else c["label"]
    return cat_id


def _all_top_ids() -> set:
    return {c["id"] for c in CATEGORY_TREE}


def _payload_for_step(state: Dict[str, Any]) -> Dict[str, Any]:
    step = state.get("step", "GREETING")

    if step == "MENU":
        return {
            "message": "What would you like to do today?",
            "ui_type": "menu",
            "options": MENU_ACTIONS,
            "ready": False,
            "agent_prompt": None,
        }

    if step == "COLLECT_CATEGORY":
        return {
            "message": "Select the **main category** of your product:",
            "ui_type": "categories",
            "options": _top_level_options(),
            "ready": False,
            "agent_prompt": None,
        }

    if step == "COLLECT_SUBCATEGORY":
        cat_id = state.get("category", "")
        cat_label = _category_label(cat_id, None)
        return {
            "message": f"Great — **{cat_label}**! Now pick a sub-category:",
            "ui_type": "categories",
            "options": _subcategory_options(cat_id),
            "ready": False,
            "agent_prompt": None,
        }

    if step == "COLLECT_PRODUCT":
        cat_label = _category_label(state.get("category", ""), state.get("subcategory"))
        return {
            "message": (
                f"Perfect — **{cat_label}**!\n\n"
                "Describe your product:\n"
                "• Name / model\n"
                "• Key features (material, size, color, specs)\n"
                "• What makes it unique"
            ),
            "ui_type": "input",
            "options": None,
            "ready": False,
            "agent_prompt": None,
        }

    if step == "COLLECT_CONSTRAINTS":
        return {
            "message": (
                "Any constraints or preferences?\n\n"
                "Examples:\n"
                "• Target audience\n"
                "• Tone\n"
                "• Language (default: English)\n"
                "• Things to avoid (e.g. no emojis)\n\n"
                "Or type **skip**."
            ),
            "ui_type": "input",
            "options": None,
            "ready": False,
            "agent_prompt": None,
        }

    return {
        "message": "Something went wrong. Let's start over!",
        "ui_type": "menu",
        "options": MENU_ACTIONS,
        "ready": False,
        "agent_prompt": None,
    }


def _go_back(state: Dict[str, Any]) -> Dict[str, Any]:
    step = state.get("step", "GREETING")

    if step in ("GREETING", "MENU"):
        return {**_empty_state(), "step": "MENU"}

    if step == "COLLECT_CATEGORY":
        return {**_empty_state(), "step": "MENU"}

    if step == "COLLECT_SUBCATEGORY":
        return {**state, "step": "COLLECT_CATEGORY", "subcategory": None}

    if step == "COLLECT_PRODUCT":
        cat_id = state.get("category")
        if cat_id and _has_subcategories(cat_id):
            return {**state, "step": "COLLECT_SUBCATEGORY", "product": None, "constraints": None}
        return {**state, "step": "COLLECT_CATEGORY", "category": None, "subcategory": None, "product": None, "constraints": None}

    if step == "COLLECT_CONSTRAINTS":
        return {**state, "step": "COLLECT_PRODUCT", "constraints": None}

    if step == "GENERATE":
        return {**state, "step": "COLLECT_CONSTRAINTS"}

    return {**_empty_state(), "step": "MENU"}


def process_message(user_input: str, state: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if state is None or not state.get("step"):
        state = _empty_state()

    step = state.get("step", "GREETING")
    text = (user_input or "").strip()

    # Back command handled globally
    if text and text.lower() in BACK_COMMANDS:
        new_state = _go_back(state)
        return new_state, _payload_for_step(new_state)

    # GREETING
    if step == "GREETING":
        new_state = {**state, "step": "MENU"}
        return new_state, {
            "message": (
                "👋 Welcome to **Ad-Wise**!\n\n"
                "I help you craft high-converting ad copy grounded in real product listing data.\n\n"
                "Choose an option:"
            ),
            "ui_type": "menu",
            "options": MENU_ACTIONS,
            "ready": False,
            "agent_prompt": None,
        }

    # MENU
    if step == "MENU":
        valid = {a["id"] for a in MENU_ACTIONS}
        if text not in valid:
            return state, {
                "message": "Please choose one of the options below:",
                "ui_type": "menu",
                "options": MENU_ACTIONS,
                "ready": False,
                "agent_prompt": None,
            }
        new_state = {**state, "step": "COLLECT_CATEGORY", "action": text}
        return new_state, _payload_for_step(new_state)

    # COLLECT_CATEGORY
    if step == "COLLECT_CATEGORY":
        if text not in _all_top_ids():
            return state, {
                "message": "Please select a category from the list:",
                "ui_type": "categories",
                "options": _top_level_options(),
                "ready": False,
                "agent_prompt": None,
            }

        new_state = {**state, "category": text}
        if _has_subcategories(text):
            new_state["step"] = "COLLECT_SUBCATEGORY"
            return new_state, _payload_for_step(new_state)

        new_state["step"] = "COLLECT_PRODUCT"
        new_state["subcategory"] = None
        return new_state, _payload_for_step(new_state)

    # COLLECT_SUBCATEGORY
    if step == "COLLECT_SUBCATEGORY":
        cat_id = state.get("category", "")
        valid_subs = {s["id"] for s in _subcategory_options(cat_id)}
        if text not in valid_subs:
            return state, {
                "message": "Please select a sub-category:",
                "ui_type": "categories",
                "options": _subcategory_options(cat_id),
                "ready": False,
                "agent_prompt": None,
            }
        new_state = {**state, "step": "COLLECT_PRODUCT", "subcategory": text}
        return new_state, _payload_for_step(new_state)

    # COLLECT_PRODUCT
    if step == "COLLECT_PRODUCT":
        if len(text) < 5:
            return state, {"message": "Please describe your product in a bit more detail:", "ui_type": "input", "options": None, "ready": False, "agent_prompt": None}
        new_state = {**state, "step": "COLLECT_CONSTRAINTS", "product": text}
        return new_state, _payload_for_step(new_state)

    # COLLECT_CONSTRAINTS -> GENERATE
    if step == "COLLECT_CONSTRAINTS":
        constraints = "" if text.lower() == "skip" else text
        new_state = {**state, "step": "GENERATE", "constraints": constraints}

        action = new_state.get("action", "full_ad")
        cat_id = new_state.get("category", "")
        sub_id = new_state.get("subcategory")
        product = new_state.get("product", "")
        cons_str = constraints if constraints else "None"
        cat_label = _category_label(cat_id, sub_id)
        rag_category = _pinecone_id(cat_id, sub_id)

        action_instruction = {
            "full_ad":       "Write a full high-converting ad: headline + 5 bullets + short description + keywords + publishing tips.",
            "headline_only": "Write ONLY a high-converting headline for this product. Output only 'Headline: ...'.",
            "keywords_5":    "Generate ONLY a list of 5 keywords/phrases that MUST be included in the headline. Output only 'Keywords: k1, k2, k3, k4, k5'.",
        }.get(action, "Write a full high-converting ad.")

        agent_prompt = (
            f"Product: {product}\n"
            f"Category: {cat_label}\n"
            f"RAG Category Filter: {rag_category}\n"
            f"Constraints: {cons_str}\n"
            f"Platform: E-commerce\n"
            f"Task: {action_instruction}"
        )

        return new_state, {
            "message": f"✅ Got it! Generating output for _{cat_label}_...",
            "ui_type": "result",
            "options": None,
            "ready": True,
            "agent_prompt": agent_prompt,
        }

    # GENERATE: after result, restart to MENU
    if step == "GENERATE":
        new_state = {**_empty_state(), "step": "MENU"}
        return new_state, _payload_for_step(new_state)

    return _empty_state(), _payload_for_step(_empty_state())