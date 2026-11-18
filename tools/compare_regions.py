#!/usr/bin/env python3
"""Compare two annotators' region boxes (pilot step A6).

Reads data/regions/annotations/<A>.json and <B>.json (written by
draw_regions.py), prints a markdown table with IoU per region, writes
data/regions/comparison_<A>_<B>.json and a side-by-side preview per sample
with both annotators' boxes drawn (A green, B orange).  The disagreement
*type* column is left blank for the annotators to fill after discussion.
"""
import argparse, json, os
import cv2
import numpy as np


def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    u = area(a) + area(b) - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", default="A"); ap.add_argument("--b", default="B")
    ap.add_argument("--dir", default="data/regions/annotations")
    ap.add_argument("--meta", default="data/OmitI2V/meta.json")
    args = ap.parse_args()

    A = json.load(open(os.path.join(args.dir, f"{args.a}.json")))
    B = json.load(open(os.path.join(args.dir, f"{args.b}.json")))
    root = os.path.dirname(args.meta)
    rows, out = [], {}
    print(f"| id | region | {args.a} | {args.b} | IoU | disagreement type | final (agreed) |")
    print("|---|---|---|---|---|---|---|")
    for sid in A:
        ea, eb = A[sid], B.get(sid)
        labels = sorted(set(ea["boxes"]) | (set(eb["boxes"]) if eb else set()))
        for lab in labels:
            ba = ea["boxes"].get(lab); bb = eb["boxes"].get(lab) if eb else None
            v = iou(ba, bb) if ba and bb else None
            rows.append((sid, lab, ba, bb, v))
            out.setdefault(sid, {})[lab] = {args.a: ba, args.b: bb, "iou": None if v is None else round(v, 3)}
            fmt = lambda b: "`" + ",".join(f"{x:.3f}" for x in b) + "`" if b else "missing"
            print(f"| {sid} | {lab} | {fmt(ba)} | {fmt(bb)} | {'' if v is None else f'{v:.2f}'} | | |")
        # side-by-side preview
        img = cv2.imread(os.path.join(root, ea["image_path"])); H, W = img.shape[:2]
        panels = []
        for who, entry, col in ((args.a, ea, (0, 200, 0)), (args.b, eb, (0, 140, 255))):
            p = img.copy()
            if entry:
                for lab, b in entry["boxes"].items():
                    x0, y0, x1, y1 = int(b[0]*W), int(b[1]*H), int(b[2]*W), int(b[3]*H)
                    cv2.rectangle(p, (x0, y0), (x1, y1), col, max(3, W // 300))
                    cv2.putText(p, lab, (x0 + 10, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)
            cv2.putText(p, who, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, col, 5)
            panels.append(cv2.resize(p, (768, int(768 * H / W))))
        both = img.copy()
        for entry, col in ((ea, (0, 200, 0)), (eb, (0, 140, 255))):
            if entry:
                for lab, b in entry["boxes"].items():
                    cv2.rectangle(both, (int(b[0]*W), int(b[1]*H)), (int(b[2]*W), int(b[3]*H)), col, max(3, W // 300))
        panels.append(cv2.resize(both, (768, int(768 * H / W))))
        cv2.imwrite(os.path.join(args.dir, f"compare_{args.a}_{args.b}_{sid}.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 88])
    vals = [r[4] for r in rows if r[4] is not None]
    print(f"\nregions compared: {len(vals)}  mean IoU {np.mean(vals):.3f}  min {min(vals):.3f}  max {max(vals):.3f}" if vals else "\nno overlapping regions")
    jp = os.path.join(args.dir, f"comparison_{args.a}_{args.b}.json")
    json.dump(out, open(jp, "w"), indent=1); print("wrote", jp, "and compare_*.jpg previews")


if __name__ == "__main__":
    main()
