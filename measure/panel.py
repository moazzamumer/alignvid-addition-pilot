#!/usr/bin/env python3
"""Step B3: one panel PNG per sample and a combined PDF.

Per sample: reference image with the intended region(s) drawn, the instruction
and expected-appearance window, then one 6-frame strip per arm (f00 f03 f06 f09
f12 f15 from outputs/<arm>/frames/<id>/) with a caption line of human labels
(from labels/labels.csv via labels/blind_map.csv) and, if present, the detector
summary from outputs/<arm>/detect/<id>.json.  Missing arms/labels are shown as
"not run" / "pending" so the panel can be built at any stage.

  python measure/panel.py --arms vanilla alignvid --out report/panels
  python measure/panel.py --ids sample_240 --arms vanilla --frames-root outputs --out report/panels"""
import argparse, csv, json, os, sys, textwrap
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
from sidecar import Sidecar

STRIP = ["f00", "f03", "f06", "f09", "f12", "f15"]
LABEL_FIELDS = ["presence", "onset_frame", "persistence", "placement", "object_correct", "preservation",
                "primary_failure", "within_window", "scorable"]
W, PAD, TILE_W, REF_W = 2000, 24, 300, 520
FONT_DIR = "/usr/share/fonts/truetype/dejavu/"


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_DIR + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)
    except OSError:
        return ImageFont.load_default()


F, FB, FS = font(22), font(26, True), font(19)


def load_labels(labels_csv, blind_map_csv):
    """-> {(arm, id): {field: value}} using the blind map video_code -> arm,id."""
    if not (labels_csv and os.path.exists(labels_csv) and blind_map_csv and os.path.exists(blind_map_csv)):
        return {}
    bm = {r["video_code"]: (r["arm"], r["id"]) for r in csv.DictReader(open(blind_map_csv))}
    out = {}
    for r in csv.DictReader(open(labels_csv)):
        if r.get("video_code") in bm:
            out.setdefault(bm[r["video_code"]], []).append(r)
    return out


def caption(rows):
    if not rows:
        return "human labels: pending"
    parts = []
    for r in rows:
        kv = "  ".join(f"{k}={r.get(k, '') or '-'}" for k in LABEL_FIELDS)
        parts.append(f"[{r.get('labeller', '?')}] {kv}")
    return "\n".join(parts)


def det_caption(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return (f"detector ('{d['query']}'): present={d['present']} ({d['n_detected']}/{d['n_frames']} frames)  "
            f"onset={d['onset_frame']}  persistence={d['persistence']}  placement={d['placement']}")


def draw_text(dr, xy, text, fnt, fill=(0, 0, 0), width_px=None):
    x, y = xy
    lines = []
    for para in text.split("\n"):
        if width_px:
            n = max(20, int(width_px / (fnt.size * 0.55)))
            lines += textwrap.wrap(para, n) or [""]
        else:
            lines.append(para)
    for ln in lines:
        dr.text((x, y), ln, font=fnt, fill=fill); y += int(fnt.size * 1.35)
    return y


def build_panel(vid, sample, meta_row, window, arms, frames_root, labels, out_path, root_img, freeze=""):
    # reference with regions
    ref = Image.open(os.path.join(root_img, sample["image_path"])).convert("RGB")
    rw, rh = ref.size; ref_s = ref.resize((REF_W, int(REF_W * rh / rw)))
    dr = ImageDraw.Draw(ref_s)
    for k, reg in sample["regions"].items():
        b = reg["final"]; box = [b[0] * ref_s.width, b[1] * ref_s.height, b[2] * ref_s.width, b[3] * ref_s.height]
        dr.rectangle(box, outline=(0, 200, 0), width=4); dr.text((box[0] + 6, box[1] + 4), k, font=FB, fill=(0, 200, 0))
    tile_h = int(TILE_W * rh / rw)  # generated frames may have another aspect; resized to tile width anyway
    n_arms = len(arms)
    row_h = tile_h + 30 + 3 * 30 + 20
    H = PAD + max(ref_s.height, 200) + PAD + n_arms * row_h + PAD
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
    im.paste(ref_s, (PAD, PAD))
    x = PAD + REF_W + PAD; y = PAD
    d.text((x, y), vid, font=FB, fill=(0, 0, 0)); y += 38
    if freeze:
        y = draw_text(d, (x, y), freeze, FS, fill=(60, 60, 60), width_px=W - x - PAD) + 4
    y = draw_text(d, (x, y), "Instruction: " + meta_row["prompt"], F, width_px=W - x - PAD)
    y = draw_text(d, (x, y + 6), "Window: " + window, FS, fill=(60, 60, 60), width_px=W - x - PAD)
    y = draw_text(d, (x, y + 6), "Regions: " + "; ".join(f"{k} [{', '.join(f'{v:.3f}' for v in r['final'])}]" for k, r in sample["regions"].items()),
                  FS, fill=(60, 60, 60), width_px=W - x - PAD)
    y = PAD + max(ref_s.height, 200) + PAD
    for arm in arms:
        d.text((PAD, y), arm, font=FB, fill=(0, 0, 0))
        fdir = os.path.join(frames_root, arm, "frames", vid)
        if os.path.isdir(fdir):
            for j, f in enumerate(STRIP):
                p = os.path.join(fdir, f + ".jpg")
                if os.path.exists(p):
                    t = Image.open(p).convert("RGB"); th = int(TILE_W * t.height / t.width)
                    im.paste(t.resize((TILE_W, th)), (PAD + j * (TILE_W + 8), y + 30))
                d.text((PAD + j * (TILE_W + 8) + 4, y + 30 + 4), f, font=FS, fill=(255, 255, 0))
        else:
            d.text((PAD, y + 30 + tile_h // 2), "not run", font=F, fill=(160, 0, 0))
        yy = y + 30 + tile_h + 6
        yy = draw_text(d, (PAD, yy), caption(labels.get((arm, vid), [])), FS, width_px=W - 2 * PAD)
        dc = det_caption(os.path.join(frames_root, arm, "detect", vid + ".json"))
        if dc:
            yy = draw_text(d, (PAD, yy), dc, FS, fill=(0, 0, 160), width_px=W - 2 * PAD)
        y += row_h
    im = im.crop((0, 0, W, min(H, y + PAD)))
    im.save(out_path); return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--arms", nargs="+", default=["vanilla", "alignvid"])
    ap.add_argument("--frames-root", default="outputs")
    ap.add_argument("--regions", default="data/regions/regions.json")
    ap.add_argument("--meta", default="data/OmitI2V/meta.json")
    ap.add_argument("--protocol", default="protocol/pilot_protocol.md", help="read §5 window per id")
    ap.add_argument("--labels", default="labels/labels.csv")
    ap.add_argument("--blind-map", default="labels/blind_map.csv")
    ap.add_argument("--out", default="report/panels")
    ap.add_argument("--pdf", default="panel.pdf", help="combined PDF name inside --out")
    a = ap.parse_args()
    regions = json.load(open(a.regions))["samples"]
    meta = {m["id"]: m for m in json.load(open(a.meta, encoding="utf-8"))}
    windows, kv = {}, {}
    if os.path.exists(a.protocol):
        for ln in open(a.protocol, encoding="utf-8"):
            if ln.startswith("| sample_") and "onset" in ln:
                c = [v.strip() for v in ln.strip().strip("|").split("|")]
                windows[c[0]] = c[1].replace("`", "")
            for key, tag in (("on", "**Frozen on:**"), ("commit", "**Frozen at commit:**")):
                if ln.startswith(tag):
                    kv[key] = ln[len(tag):].strip().strip("`")
    freeze = f"Protocol frozen {kv['on']}, commit {kv['commit']}" if kv else ""
    labels = load_labels(a.labels, a.blind_map)
    os.makedirs(a.out, exist_ok=True)
    with Sidecar(os.path.join(a.out, "panel.sidecar.json"), a) as sc:
        pages = []
        for vid in (a.ids or list(regions)):
            p = os.path.join(a.out, f"{vid}.png")
            pages.append(build_panel(vid, regions[vid], meta[vid], windows.get(vid, "n/a"), a.arms, a.frames_root, labels, p,
                                     os.path.dirname(a.meta), freeze))
            print("wrote", p)
        pdf = os.path.join(a.out, a.pdf)
        pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=150)
        sc.add(pages=len(pages), pdf=pdf, labels_found=len(labels), freeze=freeze)
        print("wrote", pdf)


if __name__ == "__main__":
    main()
