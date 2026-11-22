#!/usr/bin/env python3
"""Blind labelling kit (protocol §7). Assigns a random code to every finished video, builds one folder per
code with the video, its 16-frame contact sheet, the reference image with the intended region(s) and the
instruction + §5 window, and writes a labels CSV skeleton per labeller.

  python tools/make_blind_kit.py --arms vanilla alignvid --labellers A B [--seed 20261118]

Writes <labels-dir>/blind_map.csv (video_code,arm,id): do not open it while labelling.  Videos without a
contact sheet are extracted first with measure/extract_frames.py.  Missing videos are listed in kit.json."""
import argparse, csv, datetime, json, os, random, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["vanilla", "alignvid"])
    ap.add_argument("--ids", nargs="*", help="default: the samples in data/regions/regions.json")
    ap.add_argument("--outputs", default="outputs", help="root holding <arm>/<id>.mp4 and <arm>/frames/<id>/")
    ap.add_argument("--labels-dir", default="labels")
    ap.add_argument("--labellers", nargs="+", default=["A"])
    ap.add_argument("--seed", type=int, default=20261118, help="shuffle seed, recorded in kit.json")
    ap.add_argument("--regions", default="data/regions/regions.json")
    ap.add_argument("--meta", default="data/OmitI2V/meta.json")
    ap.add_argument("--protocol", default="protocol/pilot_protocol.md")
    ap.add_argument("--template", default="labels/labels_template.csv")
    ap.add_argument("--force", action="store_true", help="rebuild even if blind_map.csv exists")
    a = ap.parse_args()

    regions = json.load(open(a.regions))["samples"]
    meta = {m["id"]: m for m in json.load(open(a.meta, encoding="utf-8"))}
    ids = a.ids or list(regions)
    windows = {}
    for ln in open(a.protocol, encoding="utf-8"):
        if ln.startswith("| sample_") and "onset" in ln:
            c = [v.strip() for v in ln.strip().strip("|").split("|")]
            windows[c[0]] = c[1].replace("`", "")

    kit = os.path.join(a.labels_dir, "kit"); map_csv = os.path.join(a.labels_dir, "blind_map.csv")
    if os.path.exists(map_csv) and not a.force:
        sys.exit(f"{map_csv} exists; labelling may have started. Use --force only to rebuild from scratch.")
    if os.path.isdir(kit):
        shutil.rmtree(kit)
    os.makedirs(kit)

    found, missing = [], []
    for arm in a.arms:
        for vid in ids:
            mp4 = os.path.join(a.outputs, arm, f"{vid}.mp4")
            if os.path.exists(mp4):
                sheet = os.path.join(a.outputs, arm, "frames", vid, "sheet.jpg")
                if not os.path.exists(sheet):
                    subprocess.run([sys.executable, os.path.join(ROOT, "measure", "extract_frames.py"), mp4,
                                    "--out-dir", os.path.join(a.outputs, arm, "frames")], check=True)
                found.append((arm, vid, mp4, sheet))
            else:
                missing.append({"arm": arm, "id": vid, "expected": mp4})

    order = list(range(len(found))); random.Random(a.seed).shuffle(order)
    codes = {}
    for code_i, k in enumerate(order, start=1):
        arm, vid, mp4, sheet = found[k]; code = f"V{code_i:02d}"; codes[code] = (arm, vid)
        d = os.path.join(kit, code); os.makedirs(d)
        shutil.copy(mp4, os.path.join(d, f"{code}.mp4")); shutil.copy(sheet, os.path.join(d, "sheet.jpg"))
        ref_box = os.path.join(os.path.dirname(a.regions), f"{vid}_box.jpg")
        if os.path.exists(ref_box):
            shutil.copy(ref_box, os.path.join(d, "reference_with_region.jpg"))
        sample = regions[vid]
        with open(os.path.join(d, "instruction.txt"), "w", encoding="utf-8") as f:
            f.write(f"Code: {code}\nInstruction: {meta[vid]['prompt']}\n")
            f.write("Intended region(s): " + "; ".join(f"{k} [{', '.join(f'{v:.3f}' for v in r['final'])}]" for k, r in sample["regions"].items()) + "\n")
            f.write(f"Expected-appearance window (protocol §5): {windows.get(vid, 'n/a')}\n")
            f.write("Label from sheet.jpg (f00..f15); watch the video once for anything the frames miss. Rubric: protocol/label_rubric.md\n")

    with open(map_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["video_code", "arm", "id"])
        for code in sorted(codes):
            w.writerow([code, *codes[code]])
    header = next(csv.reader(open(a.template)))
    for lab in a.labellers:
        out = os.path.join(a.labels_dir, f"labels_{lab}.csv")
        if os.path.exists(out) and not a.force:
            print("kept existing", out); continue
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header); w.writeheader()
            for code in sorted(codes):
                w.writerow({"video_code": code, "labeller": lab})
    shutil.copy(os.path.join(ROOT, "protocol", "label_rubric.md"), os.path.join(kit, "label_rubric.md"))
    with open(os.path.join(kit, "README.txt"), "w") as f:
        f.write("One folder per coded video: <code>.mp4, sheet.jpg (16 frames f00..f15), reference_with_region.jpg, instruction.txt.\n"
                "Fill one row per code in labels_<you>.csv following label_rubric.md. Do not open blind_map.csv.\n")
    rec = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "git": git_hash(), "seed": a.seed, "arms": a.arms,
           "ids": ids, "n_coded": len(codes), "missing": missing, "labellers": a.labellers}
    json.dump(rec, open(os.path.join(kit, "kit.json"), "w"), indent=1)
    print(f"{len(codes)} videos coded -> {kit}; map {map_csv}; missing {len(missing)}")
    for m in missing:
        print("  missing", m["arm"], m["id"])


if __name__ == "__main__":
    main()
