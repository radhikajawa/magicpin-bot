# Vera Challenge Submission 

## Approach

`bot.py` is a single-file FastAPI server implementing all 5 endpoints from
`challenge-testing-brief.md`. The composer is a **single structured prompt**
(`SYSTEM_PROMPT` in `bot.py`) that receives the four context layers as JSON
and a short per-`trigger.kind` framing hint (`TRIGGER_FRAMING`), rather than
maintaining a separate prompt template per trigger kind — this keeps the
rulebook (specificity, voice match, single CTA, anti-fabrication, etc.) in
one place while still nudging structure per trigger type.

**Deterministic**: LLM calls are made with `temperature=0` as required.

**Graceful degradation**: if no `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` is set,
or the LLM call fails/times out, `template_compose()` produces a
context-aware fallback message (still pulling real numbers/offers/digest
items out of the contexts, never generic boilerplate) so the bot never
returns an empty or malformed action. This also means local dev works
without burning API budget — you only need a key for the qualitative
"good" tier of copy.

**Post-LLM validation** (`validate_and_fix`): strips/rejects URLs (hard
fail per testing-brief §"Body too long"/"URL in body"), rejects empty
bodies, and enforces anti-repetition within a conversation (falls back to
the template composer, and as a last resort appends an honest "following
up on this" rather than silently re-sending byte-identical text).

**Conversation state machine** (`handle_reply`), covering the three replay
scenarios called out in the brief:
- **Auto-reply detection**: same normalized merchant message twice in a
  row → `wait` 24h; three times → `end`.
- **Intent transition**: regex over common commit phrases (English +
  Hindi-English) switches the conversation into "action mode" and forces
  a `binary_confirm_cancel` CTA instead of another qualifying question.
- **Hostile / opt-out**: keyword match → graceful `end`.
- **Off-topic curveball**: short redirect back to the mission without
  ignoring the merchant.

**Suppression**: a global `SUPPRESSED` set dedupes on `suppression_key` so
the same trigger never fires twice across ticks (restraint is rewarded
per the rubric).

## What's included

- `bot.py` — the server + composer + state machine (the actual submission).
- `conversation_handlers.py` — thin `respond(state, message, ...)` wrapper
  around the same state machine, for the optional §7.4 deliverable /
  unit testing without HTTP.
- `generate_submission.py` — offline script that loads `dataset/` +
  `test_pairs.json` (from magicpin's `generate_dataset.py`) and calls
  `compose_message()` directly to produce `submission.jsonl`.

## Tradeoffs

- **One prompt, not N prompt templates.** Faster to keep consistent and
  debug, at the cost of some per-trigger-kind nuance a bespoke template
  could add. Mitigated with `TRIGGER_FRAMING` hints.
- **Global in-memory suppression**, not per-TTL. Simpler, matches the
  60-minute test window; a production version would expire suppression
  keys instead of blocking forever.
- **Template fallback is deterministic and non-conversational** — it
  doesn't read `conversation_history`, so on a reply-turn it can produce
  the same message it already sent (caught by the anti-repetition guard,
  which then appends a short variation rather than silently repeating).
  This only shows up when no LLM key is configured; with a key, the LLM
  path is what's actually used and does read conversation history.
- **Hostile/intent detection is regex-based**, not model-based, to
  guarantee sub-second, zero-cost decisions within the 30s budget before
  any LLM call is even attempted. Trade-off: it will miss hostility or
  intent phrased in ways not covered by the pattern list.

## What additional context would have helped most

- A larger, labeled bank of real (anonymized) merchant replies per
  category, to tune the intent/hostile/auto-reply regexes beyond the
  patterns explicit in the brief.
- Per-category examples of "good" vs "flat" copy beyond the one dentist
  case study, especially for restaurants/gyms/pharmacies where the
  voice/taboo lines are less intuitively guessable than "clinical-peer."

## Running it

```bash
pip install fastapi uvicorn httpx
export ANTHROPIC_API_KEY=...        # optional
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Self-test with magicpin's harness:
```bash
export BOT_URL=http://localhost:8080
python judge_simulator.py
```

Build `submission.jsonl` offline:
```bash
python generate_dataset.py --seed-dir . --out dataset
python generate_submission.py --dataset dataset --out submission.jsonl
```
