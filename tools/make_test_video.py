#!/usr/bin/env python3
"""Make a short test video from a still image by panning a crop across it (Ken
Burns style), so the measurement scripts can be tested without any GPU.  Not part
of the pilot pipeline.  Default: 6 s at 30 fps, 640 px wide, from a dev image.

  python tools/make_test_video.py data/OmitI2V/<image> --out outputs/test/pan_test.mp4"""
import argparse, os
import cv2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--zoom", type=float, default=0.75, help="crop fraction of the image at the start")
    a = ap.parse_args()
    img = cv2.imread(a.image)
    H, W = img.shape[:2]
    cw, ch = int(W * a.zoom), int(H * a.zoom)
    out_h = int(round(a.width * ch / cw)); out_h -= out_h % 2
    n = int(round(a.seconds * a.fps))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (a.width, out_h))
    for i in range(n):
        t = i / max(n - 1, 1)
        x0 = int(t * (W - cw)); y0 = int((1 - t) * (H - ch) * 0.5)
        crop = img[y0:y0 + ch, x0:x0 + cw]
        fr = cv2.resize(crop, (a.width, out_h))
        cv2.putText(fr, f"{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # frame counter for checks
        wr.write(fr)
    wr.release()
    print(f"wrote {a.out}: {n} frames, {a.width}x{out_h}, {a.fps} fps")


if __name__ == "__main__":
    main()
