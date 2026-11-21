#!/usr/bin/env python3
"""Write data/pilot5.json (the five pilot records copied from meta.json) and
data/noedit5.json (same records with the prompt replaced by the §3.3 motion-neutral
caption from the frozen protocol).  Ids come from data/regions/regions.json."""
import argparse, json, re


def captions(protocol_path):
    out, inside = {}, False
    for ln in open(protocol_path, encoding="utf-8"):
        if ln.startswith("### 3.3"):
            inside = True
        elif inside and ln.startswith("## "):
            break
        elif inside and ln.startswith("| sample_"):
            c = [v.strip() for v in ln.strip().strip("|").split("|")]
            out[c[0]] = c[1]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", default="data/OmitI2V/meta.json")
    ap.add_argument("--regions", default="data/regions/regions.json")
    ap.add_argument("--protocol", default="protocol/pilot_protocol.md")
    a = ap.parse_args()
    ids = list(json.load(open(a.regions))["samples"])
    meta = {m["id"]: m for m in json.load(open(a.meta, encoding="utf-8"))}
    pilot = [meta[i] for i in ids]
    caps = captions(a.protocol)
    noedit = [{**meta[i], "prompt": caps[i], "original_prompt": meta[i]["prompt"], "arm": "noedit"} for i in ids]
    json.dump(pilot, open("data/pilot5.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(noedit, open("data/noedit5.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("pilot5:", [p["id"] for p in pilot]); print("noedit5:", {n["id"]: n["prompt"][:40] + "..." for n in noedit})


if __name__ == "__main__":
    main()
