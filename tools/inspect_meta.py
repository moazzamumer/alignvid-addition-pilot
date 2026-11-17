#!/usr/bin/env python3
"""Inspect the OmitI2V meta.json and index the addition/object subset.

Prints: total addition count, counts per sub-category, and for the
addition/object sub-category one line per sample.  Writes a CSV index
used for sample selection (step A5).  Standard library only.
"""
import argparse
import collections
import csv
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", default="data/OmitI2V/meta.json")
    ap.add_argument("--out", default="data/addition_object_index.csv")
    args = ap.parse_args()

    with open(args.meta, encoding="utf-8") as f:
        recs = json.load(f)
    print(f"meta.json: {len(recs)} records")

    add = [r for r in recs if r["main-category"] == "addition"]
    print(f"addition: {len(add)}")
    for k, v in sorted(collections.Counter(r["sub-category"] for r in add).items()):
        print(f"  sub-category {k}: {v}")

    obj = [r for r in add if r["sub-category"] == "object"]
    print(f"\naddition/object: {len(obj)}")
    for k, v in sorted(collections.Counter(r["domain"] for r in obj).items()):
        print(f"  domain {k}: {v}")
    for k, v in sorted(collections.Counter(r["type"] for r in obj).items()):
        print(f"  image type {k}: {v}")

    rows = []
    img_root = os.path.dirname(args.meta)
    for r in obj:
        qcats = collections.Counter(q["category"] for q in r["questions"])
        rows.append({
            "id": r["id"],
            "domain": r["domain"],
            "image_type": r["type"],
            "resolution": "x".join(map(str, r["resolution"])),
            "image_path": r["image-path"],
            "image_exists": os.path.exists(os.path.join(img_root, r["image-path"])),
            "prompt": r["prompt"],
            "key": " | ".join(r["key"]),
            "expected_change": r["expected-change"],
            "n_questions": len(r["questions"]),
            "n_expected_yes": sum(q["expected_answer"] == "yes" for q in r["questions"]),
            "question_categories": "; ".join(f"{k}:{v}" for k, v in sorted(qcats.items())),
        })

    rows.sort(key=lambda x: (x["domain"], x["id"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    missing = [r["id"] for r in rows if not r["image_exists"]]
    print(f"images missing on disk: {len(missing)} {missing if missing else ''}")

    print("\nid | domain | prompt | key | nQ | question categories")
    for r in rows:
        print(f"{r['id']} | {r['domain']} | {r['prompt']} | {r['key']} | "
              f"{r['n_questions']} | {r['question_categories']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
