#!/usr/bin/env python3
"""Step B1: extract N (default 16) uniformly spaced frames from each video to
<out-dir>/<id>/f00..f15.jpg, write a 4x4 contact sheet sheet.jpg, and a JSON
with fps, frame count, duration and the chosen frame indices.

Frame i is taken at index round(i * (F-1) / (N-1)) for i in 0..N-1, so f00 is the
first decoded frame and f15 the last.  Frames are decoded sequentially (not by
seeking) so the indices are exact on every codec.

  python measure/extract_frames.py outputs/vanilla/sample_240.mp4 --out-dir outputs/vanilla/frames
  python measure/extract_frames.py outputs/vanilla/*.mp4 --out-dir outputs/vanilla/frames"""
import argparse, json, math, os, sys
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from sidecar import Sidecar


def frame_indices(n_frames, n):
    if n_frames <= 0:
        return []
    if n_frames <= n:
        return list(range(n_frames))
    return [round(i * (n_frames - 1) / (n - 1)) for i in range(n)]


def extract(video, out_dir, n=16, cols=4, tile=320, quality=95):
    vid = os.path.splitext(os.path.basename(video))[0]
    dst = os.path.join(out_dir, vid)
    os.makedirs(dst, exist_ok=True)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:  # decode everything: metadata frame counts are unreliable
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video}")
    idx = frame_indices(len(frames), n)
    h, w = frames[0].shape[:2]
    paths = []
    for i, k in enumerate(idx):
        p = os.path.join(dst, f"f{i:02d}.jpg")
        cv2.imwrite(p, frames[k], [cv2.IMWRITE_JPEG_QUALITY, quality])
        paths.append(p)
    # contact sheet: cols x rows, each tile labelled fNN (frame k / t s)
    th = int(round(tile * h / w)); rows = math.ceil(len(idx) / cols)
    import numpy as np
    sheet = np.full((rows * (th + 24), cols * tile, 3), 255, np.uint8)
    for i, k in enumerate(idx):
        r, c = divmod(i, cols); y, x = r * (th + 24), c * tile
        sheet[y + 24:y + 24 + th, x:x + tile] = cv2.resize(frames[k], (tile, th))
        t = k / fps if fps else float("nan")
        cv2.putText(sheet, f"f{i:02d}  frame {k}  {t:.2f}s", (x + 4, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(dst, "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    info = {"video": video, "id": vid, "width": w, "height": h, "fps": round(fps, 3), "frame_count_decoded": len(frames),
            "frame_count_reported": reported, "duration_s": round(len(frames) / fps, 3) if fps else None,
            "n_extracted": len(idx), "frame_indices": idx, "frame_times_s": [round(k / fps, 3) for k in idx] if fps else None,
            "frames": [os.path.basename(p) for p in paths]}
    json.dump(info, open(os.path.join(dst, "frames.json"), "w"), indent=1)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out-dir", required=True, help="e.g. outputs/vanilla/frames")
    ap.add_argument("-n", type=int, default=16)
    ap.add_argument("--tile", type=int, default=320, help="contact-sheet tile width px")
    a = ap.parse_args()
    with Sidecar(os.path.join(a.out_dir, "extract_frames.sidecar.json"), a) as sc:
        results = {}
        for v in a.videos:
            info = extract(v, a.out_dir, a.n, tile=a.tile)
            results[info["id"]] = {k: info[k] for k in ("fps", "frame_count_decoded", "duration_s", "width", "height", "frame_indices")}
            print(f"{info['id']}: {info['width']}x{info['height']} {info['fps']} fps, {info['frame_count_decoded']} frames "
                  f"({info['duration_s']} s) -> {info['n_extracted']} frames at {info['frame_indices']}")
        sc.add(videos=results)


if __name__ == "__main__":
    main()
