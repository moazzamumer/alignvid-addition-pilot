#!/usr/bin/env python3
"""Write the final intended regions (pilot step A6) to data/regions/regions.json
and draw the previews data/regions/<id>_box.jpg.

Input: a JSON file of final boxes and disagreement notes, e.g. data/regions/final_boxes.json:
{
  "sample_240": {"boxes": {"A": [x0,y0,x1,y1], "B": [...]},
                 "disagreement": {"A": "one phrase", "B": "one phrase"},
                 "free_placement": false},
  ...
}
Also merges annotator A/B boxes and IoU from comparison_A_B.json so one file
holds the full record.  Prints the §4.2 table.
"""
import argparse, datetime, json, os
import cv2

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", default="data/regions/final_boxes.json")
    ap.add_argument("--comparison", default="data/regions/annotations/comparison_A_B.json")
    ap.add_argument("--meta", default="data/OmitI2V/meta.json")
    ap.add_argument("--out", default="data/regions/regions.json")
    a = ap.parse_args()
    final = json.load(open(a.final)); comp = json.load(open(a.comparison))
    meta = {r["id"]: r for r in json.load(open(a.meta, encoding="utf-8"))}
    root = os.path.dirname(a.meta)
    out = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
           "placement_rule": "in if >=50% of object box area inside union of regions; partial if 10-50%; out otherwise",
           "coords": "normalised [x0,y0,x1,y1], origin top-left, on the original reference image",
           "samples": {}}
    print("| id | region | annotator A | annotator B | IoU | disagreement type | final |\n|---|---|---|---|---|---|---|")
    f = lambda b: "`" + ",".join(f"{x:.3f}" for x in b) + "`" if b else "—"
    for sid, e in final.items():
        r = meta[sid]; W, H = r["resolution"]
        regs = {}
        for lab, box in e["boxes"].items():
            c = comp.get(sid, {}).get(lab, {})
            regs[lab] = {"final": box, "A": c.get("A"), "B": c.get("B"), "iou_AB": c.get("iou"),
                         "disagreement": e.get("disagreement", {}).get(lab, "")}
            print(f"| {sid} | {lab} | {f(c.get('A'))} | {f(c.get('B'))} | {c.get('iou', '')} | {regs[lab]['disagreement']} | {f(box)} |")
        out["samples"][sid] = {"image_path": r["image-path"], "image_size": [W, H], "prompt": r["prompt"],
                               "free_placement": e.get("free_placement", False), "regions": regs}
        img = cv2.imread(os.path.join(root, r["image-path"]))
        for lab, box in e["boxes"].items():
            x0, y0, x1, y1 = int(box[0]*W), int(box[1]*H), int(box[2]*W), int(box[3]*H)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), max(3, W // 300))
            cv2.putText(img, lab, (x0 + 10, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        cv2.imwrite(os.path.join(os.path.dirname(a.out), f"{sid}_box.jpg"), cv2.resize(img, (512, int(512*H/W))), [cv2.IMWRITE_JPEG_QUALITY, 90])
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out, "and <id>_box.jpg previews")

if __name__ == "__main__":
    main()
