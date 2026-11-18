#!/usr/bin/env python3
"""Step A9: choose Grounding DINO thresholds on development images that are NOT
the five pilot samples.  For each dev image we run one query for an object that
is visibly present and one for the object the image's own prompt would add,
which is absent.  Two threshold pairs from the plan.  Writes JSON + previews to
data/detector_dev/.  Nothing here touches the pilot samples or any video."""
import argparse, datetime, json, os, platform, sys, time
import cv2, torch, transformers
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from gdino import GDino

DEV = [  # (id, present query, absent query = the prompt's target object)
    ("sample_237", "boat", "stack of stones"),
    ("sample_239", "tram", "cable car"),
    ("sample_102", "mushroom", "cactus"),
]
PAIRS = [(0.35, 0.25), (0.45, 0.35)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", default="data/OmitI2V/meta.json")
    ap.add_argument("--out", default="data/detector_dev")
    ap.add_argument("--model", default=None, help="HF id or local folder; default env GDINO_MODEL or the tiny model")
    a = ap.parse_args()
    meta = {r["id"]: r for r in json.load(open(a.meta, encoding="utf-8"))}
    root = os.path.dirname(a.meta); os.makedirs(a.out, exist_ok=True)
    t0 = time.time(); g = GDino(a.model) if a.model else GDino(); load_s = time.time() - t0
    results, log = {}, []
    for sid, present, absent in DEV:
        img = Image.open(os.path.join(root, meta[sid]["image-path"])).convert("RGB")
        results[sid] = {"image_path": meta[sid]["image-path"], "size": img.size, "runs": []}
        for q, truth in ((present, "present"), (absent, "absent")):
            for bt, tt in PAIRS:
                t = time.time(); dets = g.detect(img, q, bt, tt); dt = time.time() - t
                results[sid]["runs"].append({"query": q, "truth": truth, "box_thr": bt, "text_thr": tt,
                                             "n": len(dets), "max_score": max([d["score"] for d in dets], default=None),
                                             "dets": dets, "seconds": round(dt, 2)})
                log.append(f"{sid:10s} {truth:7s} q='{q}' thr=({bt},{tt}) -> {len(dets)} boxes "
                           f"max={max([d['score'] for d in dets], default=0):.2f} {dt:.1f}s")
                print(log[-1])
                # preview
                im = cv2.cvtColor(__import__('numpy').array(img), cv2.COLOR_RGB2BGR)
                for d in dets:
                    x0, y0, x1, y1 = map(int, d["box"])
                    cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 4)
                    cv2.putText(im, f"{d['score']:.2f}", (x0 + 4, max(y0 - 6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.putText(im, f"{sid} '{q}' ({truth}) thr {bt}/{tt}: {len(dets)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 3)
                cv2.imwrite(os.path.join(a.out, f"{sid}_{truth}_{bt}_{tt}.jpg"), cv2.resize(im, (960, int(960 * img.size[1] / img.size[0]))))
    side = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "model": g.model_id, "device": g.device,
            "model_load_seconds": round(load_s, 1), "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "pairs": PAIRS, "dev": DEV, "results": results}
    json.dump(side, open(os.path.join(a.out, "dev_thresholds.json"), "w"), indent=1)
    open(os.path.join(a.out, "dev_thresholds.log"), "w").write("\n".join(log) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
