#!/usr/bin/env python3
"""Step B2: Grounding DINO over the 16 extracted frames of one or more videos,
with the frozen settings (protocol §8): box/text thresholds 0.45/0.35, presence =
detected in >= k=4 of 16 frames, onset = first detected frame, persistence =
detected frames / frames from onset to f15, placement = highest-scoring box vs.
the union of the §4 regions (in >= 50 % of box area, partial 10-50 %, out < 10 %),
pre-existing check = boxes with IoU >= 0.5 to a box found by the same query on
the reference image are flagged and not counted.

Regions are stored normalised on the original reference image.  FramePack
resizes the image (scale = max(Wg/Wo, Hg/Ho)) and center-crops to the bucket, so
each region is re-projected into the generated frame from the frame's actual size
and the reference image size (`image_size` in regions.json).

  python measure/detect.py --arm vanilla --ids sample_240 sample_188
  python measure/detect.py --frames outputs/test/frames/pan_test --query boat --ref data/OmitI2V/<img> \
         --regions-norm 0.1,0.1,0.5,0.5 --out outputs/test/detect
Outputs: <out>/<id>.json, <out>/<id>_boxes.jpg (4x4 sheet with regions + boxes), sidecar."""
import argparse, json, math, os, sys
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from sidecar import Sidecar

BOX_THR, TEXT_THR, K, PRE_IOU = 0.45, 0.35, 4, 0.5
IN_FRAC, PARTIAL_FRAC = 0.50, 0.10


def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def crop_map(orig_wh, frame_wh):
    """Pixel transform original -> generated frame for FramePack's resize + center crop."""
    Wo, Ho = orig_wh; Wg, Hg = frame_wh
    s = max(Wg / Wo, Hg / Ho)
    rw, rh = int(round(Wo * s)), int(round(Ho * s))
    return {"scale": s, "resized": [rw, rh], "left": (rw - Wg) / 2, "top": (rh - Hg) / 2, "frame": [Wg, Hg], "orig": [Wo, Ho]}


def project_region(norm_box, m):
    """Normalised box on the original -> pixel box in the generated frame (clipped)."""
    Wo, Ho = m["orig"]; Wg, Hg = m["frame"]
    x0 = norm_box[0] * Wo * m["scale"] - m["left"]; x1 = norm_box[2] * Wo * m["scale"] - m["left"]
    y0 = norm_box[1] * Ho * m["scale"] - m["top"]; y1 = norm_box[3] * Ho * m["scale"] - m["top"]
    return [max(0.0, min(Wg, x0)), max(0.0, min(Hg, y0)), max(0.0, min(Wg, x1)), max(0.0, min(Hg, y1))]


def project_pixel_box(box, m):
    s = m["scale"]; Wg, Hg = m["frame"]
    b = [box[0] * s - m["left"], box[1] * s - m["top"], box[2] * s - m["left"], box[3] * s - m["top"]]
    return [max(0.0, min(Wg, b[0])), max(0.0, min(Hg, b[1])), max(0.0, min(Wg, b[2])), max(0.0, min(Hg, b[3]))]


def frac_inside_union(box, regions, frame_wh):
    """Fraction of the box's area inside the union of region boxes (pixel mask)."""
    Wg, Hg = frame_wh
    mask = np.zeros((Hg, Wg), np.uint8)
    for r in regions:
        mask[int(r[1]):int(math.ceil(r[3])), int(r[0]):int(math.ceil(r[2]))] = 1
    x0, y0, x1, y1 = int(box[0]), int(box[1]), int(math.ceil(box[2])), int(math.ceil(box[3]))
    area = max(1, (x1 - x0) * (y1 - y0))
    return float(mask[y0:y1, x0:x1].sum()) / area


def placement_label(frac):
    return "in" if frac >= IN_FRAC else "partial" if frac >= PARTIAL_FRAC else "out"


def run_one(g, frames_dir, query, ref_path, regions_norm, orig_wh, out_dir, vid, free_placement=False):
    fr_paths = sorted(p for p in os.listdir(frames_dir) if p.startswith("f") and p.endswith(".jpg"))
    frames = [Image.open(os.path.join(frames_dir, p)).convert("RGB") for p in fr_paths]
    n = len(frames); Wg, Hg = frames[0].size
    m = crop_map(orig_wh, (Wg, Hg))
    regions_px = {k: project_region(v, m) for k, v in regions_norm.items()}
    # pre-existing check on the reference image, boxes mapped into the frame
    ref_dets = []
    if ref_path:
        ref_img = Image.open(ref_path).convert("RGB")
        for d in g.detect(ref_img, query, BOX_THR, TEXT_THR):
            d["box_in_frame"] = project_pixel_box(d["box"], m); ref_dets.append(d)
    per_frame, detected = [], []
    for i, im in enumerate(frames):
        dets = g.detect(im, query, BOX_THR, TEXT_THR)
        for d in dets:
            d["pre_existing"] = any(iou(d["box"], r["box_in_frame"]) >= PRE_IOU for r in ref_dets)
            d["frac_in_region"] = round(frac_inside_union(d["box"], list(regions_px.values()), (Wg, Hg)), 3) if regions_px else None
        counted = [d for d in dets if not d["pre_existing"]]
        per_frame.append({"frame": f"f{i:02d}", "dets": dets, "n_counted": len(counted)})
        detected.append(bool(counted))
    n_det = sum(detected)
    onset = next((i for i, d in enumerate(detected) if d), None)
    persistence = round(n_det / (n - onset), 3) if onset is not None else None
    best = None
    for pf in per_frame:
        for d in pf["dets"]:
            if not d["pre_existing"] and (best is None or d["score"] > best["score"]):
                best = {**d, "frame": pf["frame"]}
    placement = None
    if best is not None and regions_px and not free_placement:
        placement = placement_label(best["frac_in_region"])
    res = {"id": vid, "query": query, "frames_dir": frames_dir, "n_frames": n, "frame_size": [Wg, Hg],
           "thresholds": {"box": BOX_THR, "text": TEXT_THR, "k": K, "pre_existing_iou": PRE_IOU},
           "crop_map": m, "regions_norm_original": regions_norm, "regions_px_frame": regions_px, "free_placement": free_placement,
           "reference_image": ref_path, "reference_dets": ref_dets,
           "detected_frames": [f"f{i:02d}" for i, d in enumerate(detected) if d], "n_detected": n_det,
           "present": n_det >= K, "onset_frame": f"f{onset:02d}" if onset is not None else None, "onset_index": onset,
           "persistence": persistence, "best_box": best, "placement": placement, "per_frame": per_frame}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, f"{vid}.json"), "w"), indent=1)
    draw_sheet(frames, per_frame, regions_px, best, os.path.join(out_dir, f"{vid}_boxes.jpg"), f"{vid} '{query}'")
    return res


def draw_sheet(frames, per_frame, regions_px, best, path, title, cols=4, tile=400):
    w, h = frames[0].size; th = int(round(tile * h / w)); rows = math.ceil(len(frames) / cols)
    sheet = np.full((rows * (th + 22) + 28, cols * tile, 3), 255, np.uint8)
    cv2.putText(sheet, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    sx, sy = tile / w, th / h
    for i, (im, pf) in enumerate(zip(frames, per_frame)):
        r, c = divmod(i, cols); y, x = 28 + r * (th + 22), c * tile
        t = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR); t = cv2.resize(t, (tile, th))
        for k, rb in regions_px.items():
            cv2.rectangle(t, (int(rb[0] * sx), int(rb[1] * sy)), (int(rb[2] * sx), int(rb[3] * sy)), (0, 200, 0), 2)
            cv2.putText(t, k, (int(rb[0] * sx) + 3, int(rb[1] * sy) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        for d in pf["dets"]:
            col = (160, 160, 160) if d["pre_existing"] else (0, 0, 255)
            b = d["box"]; cv2.rectangle(t, (int(b[0] * sx), int(b[1] * sy)), (int(b[2] * sx), int(b[3] * sy)), col, 2)
            cv2.putText(t, f"{d['score']:.2f}" + (" pre" if d["pre_existing"] else ""), (int(b[0] * sx) + 3, max(int(b[1] * sy) - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        sheet[y + 22:y + 22 + th, x:x + tile] = t
        lab = f"{pf['frame']}  counted={pf['n_counted']}" + ("  BEST" if best and best["frame"] == pf["frame"] else "")
        cv2.putText(sheet, lab, (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", help="vanilla | alignvid | noedit: frames under outputs/<arm>/frames/<id>, output outputs/<arm>/detect")
    ap.add_argument("--ids", nargs="*", help="pilot sample ids (default: all in regions.json)")
    ap.add_argument("--frames", help="single frames dir (test mode)")
    ap.add_argument("--query", help="override / test-mode query")
    ap.add_argument("--ref", help="test-mode reference image path (optional)")
    ap.add_argument("--regions-norm", help="test-mode region(s) 'x0,y0,x1,y1[;x0,y0,x1,y1]' on the reference image")
    ap.add_argument("--out", help="output dir (default outputs/<arm>/detect)")
    ap.add_argument("--regions", default="data/regions/regions.json")
    ap.add_argument("--queries", default="data/detector_queries.json")
    ap.add_argument("--meta", default="data/OmitI2V/meta.json")
    ap.add_argument("--model", default=None, help="HF id or local folder; default env GDINO_MODEL or the tiny model")
    a = ap.parse_args()
    from gdino import GDino  # import here so --help works without torch
    g = GDino(a.model) if a.model else GDino()
    out = a.out or (f"outputs/{a.arm}/detect" if a.arm else "outputs/test/detect")
    with Sidecar(os.path.join(out, "detect.sidecar.json"), a, model=g.model_id, device=g.device) as sc:
        summary = {}
        if a.frames:  # test mode
            vid = os.path.basename(os.path.normpath(a.frames))
            regs = {}
            if a.regions_norm:
                for j, s in enumerate(a.regions_norm.split(";")):
                    regs[chr(65 + j)] = [float(v) for v in s.split(",")]
            if a.ref:
                orig = Image.open(a.ref).size
            else:
                orig = Image.open(os.path.join(a.frames, "f00.jpg")).size
            r = run_one(g, a.frames, a.query, a.ref, regs, orig, out, vid)
            summary[vid] = {k: r[k] for k in ("present", "n_detected", "onset_frame", "persistence", "placement")}
        else:
            regions = json.load(open(a.regions))["samples"]
            queries = json.load(open(a.queries))["queries"]
            meta = {m["id"]: m for m in json.load(open(a.meta, encoding="utf-8"))}
            root = os.path.dirname(a.meta)
            for vid in (a.ids or list(regions)):
                fd = f"outputs/{a.arm}/frames/{vid}"
                if not os.path.isdir(fd):
                    print(f"{vid}: no frames at {fd}, skipped"); summary[vid] = {"skipped": "no frames"}; continue
                s = regions[vid]
                regs = {k: v["final"] for k, v in s["regions"].items()}
                r = run_one(g, fd, a.query or queries[vid], os.path.join(root, meta[vid]["image-path"]), regs,
                            tuple(s["image_size"]), out, vid, s.get("free_placement", False))
                summary[vid] = {k: r[k] for k in ("present", "n_detected", "onset_frame", "persistence", "placement")}
                print(vid, summary[vid])
        sc.add(summary=summary)
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
