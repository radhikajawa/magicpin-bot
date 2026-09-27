"""
Optional deliverable (challenge-brief.md §7.4): a standalone respond()
function demonstrating multi-turn handling, independent of the HTTP layer.

This simply re-exposes bot.py's conversation state machine so it can be
unit-tested or called directly, without spinning up the server.
"""

from typing import Optional
from bot import handle_reply, CONTEXTS


def respond(state: dict, merchant_message: str, merchant_id: str,
            customer_id: Optional[str] = None, trigger_id: Optional[str] = None) -> dict:
    """
    state: a dict conversation state (same shape bot.py keeps in CONVERSATIONS) —
           pass {} for a brand new conversation.
    merchant_message: the merchant's (or customer's) latest message.
    Returns: {"action": "send"|"wait"|"end", ...}
    """
    merchant = CONTEXTS.get(("merchant", merchant_id), {}).get("payload") or {}
    category = CONTEXTS.get(("category", merchant.get("category_slug")), {}).get("payload") or {}
    customer = CONTEXTS.get(("customer", customer_id), {}).get("payload") if customer_id else None
    trigger = CONTEXTS.get(("trigger", trigger_id), {}).get("payload") if trigger_id else {
        "kind": "conversation_continuation", "scope": "merchant", "urgency": 2,
        "suppression_key": "adhoc",
    }
    return handle_reply(state, merchant_message, category, merchant, trigger, customer)
