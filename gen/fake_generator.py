#!/usr/bin/env python3
"""Stand-in for inference_f1_json.py used by `run_pilot.py --dry` on a machine without a
GPU: sleeps briefly, allocates some memory, writes <id>.mp4 (copy of a test clip if
FAKE_MP4 is set, else an empty file).  Set FAKE_FAIL=<length> to simulate failure at a length."""
import argparse, json, os, shutil, sys, time

ap = argparse.ArgumentParser(); ap.add_argument("--output_dir"); ap.add_argument("--prompt_file"); ap.add_argument("--length")
ap.add_argument("--gpu_mem"); ap.add_argument("--control_scale"); a, _ = ap.parse_known_args()
print("Free VRAM: 7.5 GB, High-VRAM Mode: False", flush=True)
if os.environ.get("FAKE_FAIL") == a.length:
    print("torch.OutOfMemoryError: CUDA out of memory (simulated)"); sys.exit(1)
buf = bytearray(50 * 2**20); time.sleep(1.0)
for r in json.load(open(a.prompt_file, encoding="utf-8")):
    dst = os.path.join(a.output_dir, f"{r['id']}.mp4")
    if os.environ.get("FAKE_MP4"):
        shutil.copy(os.environ["FAKE_MP4"], dst)
    else:
        open(dst, "wb").close()
    print(f"use time 1.0  ({r['id']}, len={a.length}, gpu_mem={a.gpu_mem}, cs={a.control_scale})")
