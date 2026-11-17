#!/usr/bin/env python3
"""Make a labelled contact sheet of OmitI2V reference images for a list of ids.

usage: contact_sheet.py --ids sample_240 sample_230 ... --out sheet.jpg [--cols 4] [--tile 512]
"""
import argparse, json, os, textwrap
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--meta", default="data/OmitI2V/meta.json")
ap.add_argument("--ids", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cols", type=int, default=4)
ap.add_argument("--tile", type=int, default=512)
a = ap.parse_args()

recs = {r["id"]: r for r in json.load(open(a.meta, encoding="utf-8"))}
root = os.path.dirname(a.meta)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except OSError:
    font = ImageFont.load_default()
cap_h = 110
n = len(a.ids); cols = a.cols; rows = -(-n // cols)
sheet = Image.new("RGB", (cols * a.tile, rows * (a.tile + cap_h)), "white")
d = ImageDraw.Draw(sheet)
for i, sid in enumerate(a.ids):
    r = recs[sid]
    im = Image.open(os.path.join(root, r["image-path"])).convert("RGB")
    im.thumbnail((a.tile, a.tile))
    x = (i % cols) * a.tile; y = (i // cols) * (a.tile + cap_h)
    sheet.paste(im, (x + (a.tile - im.width) // 2, y + (a.tile - im.height) // 2))
    cap = f"{sid} [{r['domain']}, {r['type']}, {r['resolution'][0]}x{r['resolution'][1]}]\n" + \
          "\n".join(textwrap.wrap(r["prompt"], 68)[:4])
    d.multiline_text((x + 6, y + a.tile + 4), cap, fill="black", font=font, spacing=2)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
sheet.save(a.out, quality=88)
print("wrote", a.out, sheet.size)
