"""
Builds submission.jsonl (deliverable §7.2) by running bot.py's compose_message()
directly against the dataset — no HTTP server needed for this step.

Usage:
    python generate_dataset.py --seed-dir . --out dataset      # (magicpin's script)
    python generate_submission.py --dataset dataset --out submission.jsonl
"""

import argparse
import json
from pathlib import Path

from bot import compose_message


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset", help="Path to the expanded dataset/ directory")
    ap.add_argument("--out", default="submission.jsonl")
    args = ap.parse_args()

    ds = Path(args.dataset)
    test_pairs = load_json(ds / "test_pairs.json")["pairs"]

    categories = {p.stem: load_json(p) for p in (ds / "categories").glob("*.json")}
    merchants = {p.stem: load_json(p) for p in (ds / "merchants").glob("*.json")}
    customers = {p.stem: load_json(p) for p in (ds / "customers").glob("*.json")}
    triggers = {p.stem: load_json(p) for p in (ds / "triggers").glob("*.json")}

    lines = []
    for pair in test_pairs:
        trigger = triggers[pair["trigger_id"]]
        merchant = merchants[pair["merchant_id"]]
        category = categories[merchant["category_slug"]]
        customer = customers.get(pair["customer_id"]) if pair.get("customer_id") else None

        result = compose_message(category, merchant, trigger, customer)
        lines.append({
            "test_id": pair["test_id"],
            "body": result["body"],
            "cta": result["cta"],
            "send_as": result["send_as"],
            "suppression_key": result["suppression_key"],
            "rationale": result["rationale"],
        })
        print(f"{pair['test_id']}: {result['body'][:80]}...")

    with open(args.out, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(lines)} lines to {args.out}")


if __name__ == "__main__":
    main()
