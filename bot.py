"""
magicpin AI Challenge — "Vera, but better"
============================================
A single-file FastAPI bot implementing the 5-endpoint contract from
challenge-testing-brief.md, and the compose() logic from challenge-brief.md.

Run:
    pip install fastapi uvicorn httpx
    export ANTHROPIC_API_KEY=...        # optional — bot works without it (template fallback)
    uvicorn bot:app --host 0.0.0.0 --port 8080

Self-test:
    export BOT_URL=http://localhost:8080
    python judge_simulator.py
"""

import os
import re
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel

log = logging.getLogger("vera_bot")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Vera Challenge Bot")
START_TIME = time.time()

# ---------------------------------------------------------------------------
# In-memory state (swap for Redis/SQLite for a real deployment — spec allows
# in-memory as long as the process doesn't restart mid-test).
# ---------------------------------------------------------------------------

# (scope, context_id) -> {"version": int, "payload": dict}
CONTEXTS: dict[tuple[str, str], dict] = {}

# conversation_id -> state
CONVERSATIONS: dict[str, dict] = {}

# suppression_key -> True (already sent once; simple global dedup for this test run)
SUPPRESSED: set[str] = set()

TEAM_METADATA = {
    "team_name": os.environ.get("TEAM_NAME", "Team CrackNonTech"),
    "team_members": os.environ.get("TEAM_MEMBERS", "Jitendra Kumar").split(","),
    "model": os.environ.get("LLM_MODEL", "claude-sonnet-4-6"),
    "approach": (
        "Single structured-prompt composer keyed off trigger.kind, with a "
        "deterministic template fallback when no LLM key is configured. "
        "Post-LLM validation (URL ban, empty-body ban, anti-repetition, CTA "
        "shape) plus a small conversation state machine for auto-reply "
        "detection, intent-transition handling, and hostile/off-topic exits."
    ),
    "contact_email": os.environ.get("CONTACT_EMAIL", "team@example.com"),
    "version": "1.0.0",
    "submitted_at": datetime.now(timezone.utc).isoformat(),
}

# ---------------------------------------------------------------------------
# LLM call — pluggable, deterministic (temperature=0), with a safe fallback.
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
LLM_TIMEOUT_S = 20  # stay well inside the 30s budget the judge allows


def call_llm(system: str, user: str) -> Optional[str]:
    """Returns raw text from the LLM, or None if unavailable/failed.
    Caller must handle the None case with a template fallback — the bot
    must never hang past the tick/reply budget."""
    try:
        if LLM_PROVIDER == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            import httpx
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "max_tokens": 500,
                    "temperature": 0,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
                timeout=LLM_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()
            return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

        if LLM_PROVIDER == "openai" and os.environ.get("OPENAI_API_KEY"):
            import httpx
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json={
                    "model": LLM_MODEL or "gpt-4o-mini",
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=LLM_TIMEOUT_S,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001 — never let an LLM error break a tick
        log.warning("LLM call failed, falling back to template composer: %s", e)
        return None
    return None  # no key configured -> template fallback


def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


# ---------------------------------------------------------------------------
# The composer — this is the heart of the submission.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Vera, magicpin's AI assistant that talks to merchants (and, on their
behalf, their customers) over WhatsApp. You compose ONE outbound message given four context
layers: category, merchant, trigger, and (optionally) customer.

Hard rules:
1. Anchor on a concrete, verifiable fact from the given contexts (a number, a date, a headline,
   a peer stat). Never write generic filler like "10% off" or "grow your business" when a
   specific fact is available.
2. Match the category voice exactly (tone, vocabulary, taboos). Clinical/peer categories
   (dentists, doctors) must NOT sound promotional.
3. Personalize to this specific merchant/customer — use their real numbers, offers, signals,
   conversation history, language preference. Never invent data not present in the contexts.
4. State why you're messaging *now* — tie the message explicitly to the trigger.
5. Use at most ONE compulsion lever combo naturally: specificity, loss aversion, social proof,
   effort externalization, curiosity, reciprocity, asking the merchant a question, or a single
   binary CTA. Never stack multiple CTAs.
6. The call-to-action must be a single, clear ask, landing in the last sentence. Use
   "none" only for pure-information triggers.
7. No URLs. No long preambles ("I hope you're doing well..."). Don't re-introduce yourself if
   conversation_history is non-empty.
8. Hindi-English code-mix is fine and often preferred — match merchant/customer language
   preference.
9. If you don't have enough data to be specific, say less rather than fabricate.
10. Never fabricate research citations, competitor names, or offers not present in context.

Respond with ONLY a JSON object, no prose, no markdown fences:
{"body": "...", "cta": "binary_yes_no|open_ended|multi_choice_slot|binary_confirm_cancel|none",
 "rationale": "one or two sentences on why this message, what it should achieve"}
"""

# Trigger-kind -> short framing hint injected into the prompt. Keeps one system
# prompt but nudges tone/structure per kind (routing layer from brief §13.2).
TRIGGER_FRAMING = {
    "research_digest": "Frame around the single most merchant-relevant digest item. Cite source.",
    "regulation_change": "Frame as a compliance heads-up, calm and factual, not alarming.",
    "category_research_digest_release": "Frame around the single most relevant digest item. Cite source.",
    "perf_spike": "Frame as good news + how to capitalize on it right now.",
    "perf_dip": "Frame as loss aversion — name the specific metric drop and offer a concrete fix.",
    "milestone_reached": "Frame as a celebratory, low-pressure moment. CTA can be 'none' or light.",
    "dormant_with_vera": "Re-engage gently. Reference something concrete from their account, not a generic 'hi'.",
    "customer_lapsed_soft": "This is a recall reminder to a customer — offer real open slots and a real offer price.",
    "appointment_tomorrow": "A reminder to the customer about tomorrow's booking. Keep it short.",
    "review_theme_emerged": "Reference the specific theme reviewers raised; ask the merchant one clarifying question.",
    "scheduled_recurring": "This is a weekly curiosity-driven check-in — ask the merchant something, don't just push info.",
    "festival_upcoming": "Tie the upcoming festival to a concrete, category-correct offer or content idea.",
    "weather_heatwave": "Only useful if it plausibly changes customer behavior for this category — be concrete.",
    "local_news_event": "Only send if genuinely relevant to this merchant's footfall or operations.",
    "competitor_opened": "Frame as competitive awareness, not alarmism. Offer a concrete next step.",
    "category_trend_movement": "Tie the trend number to something the merchant could act on.",
}


def build_user_prompt(category: dict, merchant: dict, trigger: dict, customer: Optional[dict],
                       conversation_history: Optional[list[dict]] = None) -> str:
    kind = trigger.get("kind", "")
    framing = TRIGGER_FRAMING.get(kind, "Compose the most relevant, specific message you can from the contexts.")
    scope = trigger.get("scope", "merchant")
    send_as = "merchant_on_behalf" if (scope == "customer" and customer) else "vera"

    parts = [
        f"send_as = {send_as}",
        f"trigger.kind = {kind}  (urgency={trigger.get('urgency')})",
        f"Framing hint: {framing}",
        "",
        "CATEGORY CONTEXT:",
        json.dumps(category, ensure_ascii=False, indent=2)[:3000],
        "",
        "MERCHANT CONTEXT:",
        json.dumps(merchant, ensure_ascii=False, indent=2)[:3000],
        "",
        "TRIGGER CONTEXT:",
        json.dumps(trigger, ensure_ascii=False, indent=2)[:1500],
    ]
    if customer:
        parts += ["", "CUSTOMER CONTEXT:", json.dumps(customer, ensure_ascii=False, indent=2)[:1500]]
    if conversation_history:
        parts += ["", "CONVERSATION SO FAR (most recent last):",
                   json.dumps(conversation_history[-6:], ensure_ascii=False, indent=2)]
    parts += ["", "Compose the next message now. JSON only."]
    return "\n".join(parts)


URL_RE = re.compile(r"https?://\S+")


def validate_and_fix(candidate: dict, category: dict, merchant: dict, trigger: dict,
                      customer: Optional[dict], already_sent: set[str]) -> dict:
    """Post-LLM validation per brief §11 anti-patterns + testing-brief §10 penalties."""
    body = (candidate.get("body") or "").strip()
    cta = candidate.get("cta") or "none"

    if not body or URL_RE.search(body) or body in already_sent:
        # Fall back to the deterministic template composer, which is
        # guaranteed URL-free and guaranteed non-empty.
        fallback = template_compose(category, merchant, trigger, customer)
        body, cta = fallback["body"], fallback["cta"]
        # last-resort de-dupe: if even the template repeats, append a short,
        # honest variation rather than silently re-sending.
        if body in already_sent:
            body = body.rstrip(".") + " — following up on this."

    return {
        "body": body,
        "cta": cta,
        "rationale": candidate.get("rationale") or "Composed from category+merchant+trigger context.",
    }


def compose_message(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None,
                     conversation_history: Optional[list[dict]] = None,
                     already_sent: Optional[set[str]] = None) -> dict:
    already_sent = already_sent or set()
    user_prompt = build_user_prompt(category, merchant, trigger, customer, conversation_history)
    raw = call_llm(SYSTEM_PROMPT, user_prompt)
    parsed = extract_json(raw) if raw else None
    if not parsed:
        parsed = template_compose(category, merchant, trigger, customer)
    fixed = validate_and_fix(parsed, category, merchant, trigger, customer, already_sent)

    scope = trigger.get("scope", "merchant")
    send_as = "merchant_on_behalf" if (scope == "customer" and customer) else "vera"
    return {
        "body": fixed["body"],
        "cta": fixed["cta"],
        "send_as": send_as,
        "suppression_key": trigger.get("suppression_key", f"generic:{trigger.get('id','')}"),
        "rationale": fixed["rationale"],
    }


# ---------------------------------------------------------------------------
# Deterministic template fallback — used when no LLM key is set, or when the
# LLM output fails validation. Still context-aware: it pulls real numbers out
# of the contexts rather than emitting boilerplate, so the bot degrades
# gracefully instead of failing outright.
# ---------------------------------------------------------------------------

def _merchant_name(merchant: dict) -> str:
    ident = merchant.get("identity", {})
    return ident.get("owner_first_name") or ident.get("name", "there")


def _lang_is_hindi_mix(ident: dict) -> bool:
    langs = ident.get("languages") or []
    pref = ident.get("language_pref", "")
    return "hi" in langs or "hi" in pref


def template_compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> dict:
    kind = trigger.get("kind", "")
    name = _merchant_name(merchant)
    perf = merchant.get("performance", {})
    peer = category.get("peer_stats", {})
    signals = merchant.get("signals", [])
    digest = category.get("digest", [])

    if kind in ("research_digest", "category_research_digest_release") and digest:
        item = digest[0]
        title = item.get("title", "a new item")
        source = item.get("source", "")
        return {
            "body": f"{name}, {source or 'this week\u2019s digest'} flagged something relevant: {title}. "
                    f"Worth a 2-min look — want me to pull it and draft something you can share?",
            "cta": "open_ended",
        }

    if kind == "perf_dip":
        delta = perf.get("delta_7d", {}).get("calls_pct")
        metric = f"calls down {abs(delta) * 100:.0f}% week-over-week" if delta else "your numbers dipped this week"
        return {
            "body": f"{name}, noticed {metric}. Want me to check what changed and suggest one fix?",
            "cta": "open_ended",
        }

    if kind == "perf_spike":
        delta = perf.get("delta_7d", {}).get("views_pct")
        metric = f"views up {delta * 100:.0f}%" if delta else "a nice uptick"
        return {
            "body": f"{name}, {metric} vs your usual — good momentum. Want me to add an offer while attention is high?",
            "cta": "open_ended",
        }

    if kind == "milestone_reached":
        return {
            "body": f"{name}, quick one — you just crossed a review milestone on your listing. Nicely done!",
            "cta": "none",
        }

    if kind == "dormant_with_vera":
        sig = signals[0] if signals else None
        extra = f" I also noticed {sig.replace('_', ' ')}." if sig else ""
        return {
            "body": f"{name}, haven't heard from you in a bit.{extra} Want me to take a quick look at your listing?",
            "cta": "open_ended",
        }

    if kind == "customer_lapsed_soft" or (trigger.get("scope") == "customer" and customer):
        offers = merchant.get("offers", [])
        offer_title = next((o["title"] for o in offers if o.get("status") == "active"), "your next visit")
        cust_name = (customer or {}).get("identity", {}).get("name", "there")
        merchant_disp = merchant.get("identity", {}).get("name", "the clinic")
        return {
            "body": f"Hi {cust_name}, {merchant_disp} here. It's been a while since your last visit — "
                    f"{offer_title} is available. Reply YES if you'd like us to hold a slot for you.",
            "cta": "binary_yes_no",
        }

    if kind == "review_theme_emerged":
        return {
            "body": f"{name}, a few recent reviews mention the same thing. Want to see the exact quotes?",
            "cta": "open_ended",
        }

    if kind == "competitor_opened":
        return {
            "body": f"{name}, heads up — a new listing opened nearby in your category. "
                    f"Want me to check how your profile compares?",
            "cta": "open_ended",
        }

    # Generic fallback — still tries to use a real peer-comparison signal if present.
    ctr = perf.get("ctr")
    avg_ctr = peer.get("avg_ctr")
    if ctr is not None and avg_ctr:
        return {
            "body": f"{name}, your listing CTR is {ctr*100:.1f}% vs a peer average of {avg_ctr*100:.1f}%. "
                    f"Want one suggestion to close that gap?",
            "cta": "open_ended",
        }
    return {
        "body": f"{name}, quick check-in — anything about your listing you'd like help with this week?",
        "cta": "open_ended",
    }


# ---------------------------------------------------------------------------
# Conversation state machine — auto-reply detection, intent transition,
# hostile/off-topic handling. Drives /v1/reply.
# ---------------------------------------------------------------------------

INTENT_PATTERNS = re.compile(
    r"\b(let'?s do it|lets do it|go ahead|i want to join|sign me up|i'?m in|im in|"
    r"confirm|haan karo|chalo|ok let'?s|sure let'?s|start karo)\b", re.IGNORECASE)

HOSTILE_PATTERNS = re.compile(
    r"\b(stop|useless|bothering|harass|spam|don'?t message|not interested|"
    r"leave me alone|band karo|pareshan)\b", re.IGNORECASE)

OFFTOPIC_HINT = re.compile(r"\b(gst|tax|unrelated|by the way|btw)\b", re.IGNORECASE)


def normalize(msg: str) -> str:
    return re.sub(r"\s+", " ", msg.strip().lower())


def handle_reply(conv: dict, message: str, category: dict, merchant: dict,
                  trigger: dict, customer: Optional[dict]) -> dict:
    norm = normalize(message)
    history = conv.setdefault("raw_merchant_messages", [])

    # --- auto-reply detection: same message verbatim repeats ---
    repeat_count = 1
    if history and history[-1] == norm:
        repeat_count = conv.get("repeat_count", 1) + 1
    conv["repeat_count"] = repeat_count
    history.append(norm)

    if repeat_count == 2:
        return {"action": "wait", "wait_seconds": 86400,
                "rationale": "Same message twice in a row — looks like an auto-reply. Backing off 24h."}
    if repeat_count >= 3:
        return {"action": "end",
                "rationale": "Same message 3+ times — confirmed auto-reply / no real engagement. Closing."}

    # --- hostile / opt-out ---
    if HOSTILE_PATTERNS.search(message):
        return {"action": "end",
                "rationale": "Merchant signaled frustration/opt-out. Closing gracefully, suppressing further sends."}

    # --- explicit intent transition: stop qualifying, move to action ---
    if INTENT_PATTERNS.search(message):
        conv["mode"] = "action"
        result = compose_message(category, merchant, trigger, customer,
                                  conversation_history=conv.get("turns"),
                                  already_sent=conv.get("sent_bodies", set()))
        body = (f"Great — {result['body']}" if not result["body"].lower().startswith("great")
                else result["body"])
        return {"action": "send", "body": body, "cta": "binary_confirm_cancel",
                "rationale": "Merchant explicitly committed; switching from pitch mode to action mode immediately."}

    # --- off-topic curveball: answer briefly, don't derail ---
    if OFFTOPIC_HINT.search(message):
        return {"action": "send",
                "body": "That's outside what I can help with here, but happy to keep helping with your "
                        "magicpin listing — want to pick up where we left off?",
                "cta": "open_ended",
                "rationale": "Off-topic question; politely redirected back to the mission without ignoring the merchant."}

    # --- normal continuation ---
    result = compose_message(category, merchant, trigger, customer,
                              conversation_history=conv.get("turns"),
                              already_sent=conv.get("sent_bodies", set()))
    return {"action": "send", "body": result["body"], "cta": result["cta"], "rationale": result["rationale"]}


# ---------------------------------------------------------------------------
# HTTP models + endpoints
# ---------------------------------------------------------------------------

class CtxBody(BaseModel):
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: Optional[str] = None


@app.post("/v1/context")
async def push_context(body: CtxBody):
    key = (body.scope, body.context_id)
    cur = CONTEXTS.get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    CONTEXTS[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.now(timezone.utc).isoformat()}


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    for trg_id in body.available_triggers[:20]:  # respect the 20-action cap
        trg = CONTEXTS.get(("trigger", trg_id), {}).get("payload")
        if not trg:
            continue
        supp_key = trg.get("suppression_key", trg_id)
        if supp_key in SUPPRESSED:
            continue

        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")
        merchant = CONTEXTS.get(("merchant", merchant_id), {}).get("payload")
        if not merchant:
            continue
        category = CONTEXTS.get(("category", merchant.get("category_slug")), {}).get("payload")
        if not category:
            continue
        customer = CONTEXTS.get(("customer", customer_id), {}).get("payload") if customer_id else None

        result = compose_message(category, merchant, trg, customer)
        conversation_id = f"conv_{merchant_id}_{trg_id}_{uuid.uuid4().hex[:6]}"

        CONVERSATIONS[conversation_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trg_id,
            "turns": [{"from": result["send_as"], "body": result["body"]}],
            "sent_bodies": {result["body"]},
            "mode": "qualifying",
            "raw_merchant_messages": [],
            "repeat_count": 1,
        }
        SUPPRESSED.add(supp_key)

        actions.append({
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result["send_as"],
            "trigger_id": trg_id,
            "template_name": f"{result['send_as']}_{trg.get('kind','generic')}_v1",
            "template_params": [result["body"]],
            "body": result["body"],
            "cta": result["cta"],
            "suppression_key": supp_key,
            "rationale": result["rationale"],
        })
    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: Optional[str] = None
    turn_number: Optional[int] = None


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv = CONVERSATIONS.setdefault(body.conversation_id, {
        "merchant_id": body.merchant_id, "customer_id": body.customer_id,
        "turns": [], "sent_bodies": set(), "mode": "qualifying",
        "raw_merchant_messages": [], "repeat_count": 1,
    })
    conv["turns"].append({"from": body.from_role, "body": body.message})

    merchant_id = body.merchant_id or conv.get("merchant_id")
    customer_id = body.customer_id or conv.get("customer_id")
    merchant = CONTEXTS.get(("merchant", merchant_id), {}).get("payload") or {}
    category = CONTEXTS.get(("category", merchant.get("category_slug")), {}).get("payload") or {}
    customer = CONTEXTS.get(("customer", customer_id), {}).get("payload") if customer_id else None
    trigger = CONTEXTS.get(("trigger", conv.get("trigger_id")), {}).get("payload") or {
        "kind": "conversation_continuation", "scope": "merchant", "urgency": 2,
        "suppression_key": body.conversation_id,
    }

    outcome = handle_reply(conv, body.message, category, merchant, trigger, customer)

    if outcome["action"] == "send":
        conv["sent_bodies"].add(outcome["body"])
        conv["turns"].append({"from": "vera", "body": outcome["body"]})
        return {"action": "send", "body": outcome["body"], "cta": outcome.get("cta", "open_ended"),
                "rationale": outcome["rationale"]}
    if outcome["action"] == "wait":
        return {"action": "wait", "wait_seconds": outcome.get("wait_seconds", 3600),
                "rationale": outcome["rationale"]}
    return {"action": "end", "rationale": outcome["rationale"]}


@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _cid), _v in CONTEXTS.items():
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START_TIME), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return TEAM_METADATA


@app.post("/v1/teardown")
async def teardown():
    CONTEXTS.clear()
    CONVERSATIONS.clear()
    SUPPRESSED.clear()
    return {"status": "wiped"}
