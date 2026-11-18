#!/usr/bin/env python3
"""Draw intended-region boxes on OmitI2V reference images (pilot step A6).

The annotator draws from the text in protocol/pilot_protocol.md §4.3 only.
The tool reads that section to learn which region labels (A, B, ...) each
sample has, shows the description under the image, and saves normalised
boxes [x0, y0, x1, y1] (0-1, origin top-left) plus a preview image.

usage:
  draw_regions.py --annotator A            # draws the five pilot samples
  draw_regions.py --annotator B --ids sample_240 sample_188

Mouse: drag to draw the box for the highlighted region label.
Keys : a / b / c  select region label      u  undo (clear current label)
       n  next sample (all labels drawn)   p  previous sample
       f  mark sample free_placement       q  save and quit
Output: data/regions/annotations/<annotator>.json and <annotator>_<id>.jpg
"""
import argparse, datetime, json, os, re, sys, textwrap
import cv2
import numpy as np

PILOT_IDS = ["sample_240", "sample_188", "sample_208", "sample_245", "sample_116"]
COLORS = {"A": (0, 200, 0), "B": (0, 140, 255), "C": (255, 0, 200)}
MAX_W, MAX_H, PANEL_H = 1500, 820, 210


def parse_descriptions(protocol_path):
    """Return {id: {"text": str, "regions": [labels]}} from §4.3."""
    s = open(protocol_path, encoding="utf-8").read()
    start = s.index("### 4.3")
    end = s.index("\n## 5")
    sec = s[start:end]
    out = {}
    blocks = re.split(r"\n(?=\*\*sample_\d+)", sec)
    for b in blocks:
        m = re.match(r"\*\*(sample_\d+)", b)
        if not m:
            continue
        labels = re.findall(r"- Region ([A-Z]) ", b)
        out[m.group(1)] = {"text": b.strip(), "regions": labels or ["A"]}
    return out


def load_meta(meta_path):
    recs = json.load(open(meta_path, encoding="utf-8"))
    return {r["id"]: r for r in recs}, os.path.dirname(meta_path)


class Annotator:
    def __init__(self, ids, meta, root, desc, annotator, out_dir):
        self.ids, self.meta, self.root, self.desc = ids, meta, root, desc
        self.annotator, self.out_dir = annotator, out_dir
        self.json_path = os.path.join(out_dir, f"{annotator}.json")
        self.data = json.load(open(self.json_path)) if os.path.exists(self.json_path) else {}
        self.i = 0
        self.drag = None
        self.win = f"draw_regions [{annotator}]"

    # ---------- per-sample state ----------
    def load(self):
        sid = self.ids[self.i]
        r = self.meta[sid]
        self.img = cv2.imread(os.path.join(self.root, r["image-path"]))
        self.H, self.W = self.img.shape[:2]
        self.scale = min(MAX_W / self.W, MAX_H / self.H, 1.0)
        self.disp = cv2.resize(self.img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self.labels = self.desc[sid]["regions"]
        self.cur = self.labels[0]
        entry = self.data.get(sid, {})
        self.boxes = {k: v for k, v in entry.get("boxes", {}).items()}
        self.free = entry.get("free_placement", False)
        self.prompt = r["prompt"]

    def norm(self, x0, y0, x1, y1):
        s = self.scale
        x0, x1 = sorted((x0 / s / self.W, x1 / s / self.W))
        y0, y1 = sorted((y0 / s / self.H, y1 / s / self.H))
        clip = lambda v: round(min(max(v, 0.0), 1.0), 4)
        return [clip(x0), clip(y0), clip(x1), clip(y1)]

    def save_sample(self):
        sid = self.ids[self.i]
        self.data[sid] = {
            "annotator": self.annotator,
            "image_path": self.meta[sid]["image-path"],
            "image_size": [self.W, self.H],
            "free_placement": self.free,
            "boxes": self.boxes,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        os.makedirs(self.out_dir, exist_ok=True)
        json.dump(self.data, open(self.json_path, "w"), indent=1)
        prev = self.img.copy()
        for lab, b in self.boxes.items():
            x0, y0, x1, y1 = int(b[0] * self.W), int(b[1] * self.H), int(b[2] * self.W), int(b[3] * self.H)
            cv2.rectangle(prev, (x0, y0), (x1, y1), COLORS.get(lab, (0, 0, 255)), max(2, self.W // 400))
            cv2.putText(prev, lab, (x0 + 8, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, COLORS.get(lab, (0, 0, 255)), 3)
        small = cv2.resize(prev, (512, int(512 * self.H / self.W)))
        cv2.imwrite(os.path.join(self.out_dir, f"{self.annotator}_{sid}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # ---------- drawing ----------
    def frame(self):
        sid = self.ids[self.i]
        canvas = self.disp.copy()
        for lab, b in self.boxes.items():
            x0, y0 = int(b[0] * self.W * self.scale), int(b[1] * self.H * self.scale)
            x1, y1 = int(b[2] * self.W * self.scale), int(b[3] * self.H * self.scale)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), COLORS.get(lab, (0, 0, 255)), 2)
            cv2.putText(canvas, lab, (x0 + 5, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS.get(lab, (0, 0, 255)), 2)
        if self.drag and self.drag[1]:
            (x0, y0), (x1, y1) = self.drag
            cv2.rectangle(canvas, (x0, y0), (x1, y1), COLORS.get(self.cur, (0, 0, 255)), 1)
        h, w = canvas.shape[:2]
        panel = 255 * np.ones((PANEL_H, w, 3), dtype="uint8")
        status = f"[{self.i+1}/{len(self.ids)}] {sid}   labels: " + "  ".join(
            f"{'>' if l == self.cur else ' '}{l}{'*' if l in self.boxes else ''}" for l in self.labels)
        status += "   FREE_PLACEMENT" if self.free else ""
        cv2.putText(panel, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        y = 50
        for line in textwrap.wrap("PROMPT: " + self.prompt, 150)[:2]:
            cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 0, 0), 1); y += 20
        for line in textwrap.wrap(self.desc[sid]["text"].replace("**", "").replace("\n", " "), 160)[:5]:
            cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1); y += 18
        cv2.putText(panel, "drag=box  a/b=label  u=undo  n=next  p=prev  f=free  q=save+quit   (full text: protocol 4.3)",
                    (10, PANEL_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        return np.vstack([canvas, panel])

    def render(self):
        cv2.imshow(self.win, self.frame())

    def on_mouse(self, ev, x, y, flags, _):
        y = min(y, self.disp.shape[0] - 1)
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.drag = [(x, y), None]
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag:
            self.drag[1] = (x, y); self.render()
        elif ev == cv2.EVENT_LBUTTONUP and self.drag:
            (x0, y0) = self.drag[0]
            if abs(x - x0) > 5 and abs(y - y0) > 5:
                self.boxes[self.cur] = self.norm(x0, y0, x, y); self.free = False
                print(f"  {self.ids[self.i]} {self.cur}: {self.boxes[self.cur]}")
            self.drag = None; self.render()

    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self.on_mouse)
        self.load()
        print("\n".join(textwrap.wrap(self.desc[self.ids[self.i]]["text"], 100)), "\n")
        while True:
            self.render()
            k = cv2.waitKey(30) & 0xFF
            if k == 255:
                continue
            c = chr(k).lower()
            if c in "abc" and c.upper() in self.labels:
                self.cur = c.upper()
            elif c == "u":
                self.boxes.pop(self.cur, None)
            elif c == "f":
                self.free, self.boxes = True, {}
            elif c in "np":
                missing = [l for l in self.labels if l not in self.boxes]
                if c == "n" and missing and not self.free:
                    print(f"  draw region(s) {missing} first, or press f for free_placement"); continue
                self.save_sample()
                if c == "n" and self.i == len(self.ids) - 1:
                    print("all samples saved ->", self.json_path); break
                self.i += 1 if c == "n" else -1
                self.i = max(0, min(self.i, len(self.ids) - 1))
                self.load()
                print("\n" + "\n".join(textwrap.wrap(self.desc[self.ids[self.i]]["text"], 100)), "\n")
            elif c == "q":
                self.save_sample(); print("saved ->", self.json_path); break
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotator", required=True, help="short label, e.g. A or B")
    ap.add_argument("--ids", nargs="+", default=PILOT_IDS)
    ap.add_argument("--meta", default="data/OmitI2V/meta.json")
    ap.add_argument("--protocol", default="protocol/pilot_protocol.md")
    ap.add_argument("--out-dir", default="data/regions/annotations")
    ap.add_argument("--check", action="store_true", help="parse protocol and images, print labels, no GUI")
    a = ap.parse_args()
    desc = parse_descriptions(a.protocol)
    meta, root = load_meta(a.meta)
    for sid in a.ids:
        if sid not in desc:
            sys.exit(f"no §4.3 description for {sid}")
        if sid not in meta:
            sys.exit(f"{sid} not in meta.json")
    if a.check:
        for sid in a.ids:
            print(sid, "regions", desc[sid]["regions"], "image", meta[sid]["resolution"])
        return
    Annotator(a.ids, meta, root, desc, a.annotator, a.out_dir).run()


if __name__ == "__main__":
    main()
