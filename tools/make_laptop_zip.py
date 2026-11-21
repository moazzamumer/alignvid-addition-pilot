#!/usr/bin/env python3
"""Assemble the generation package for a remote machine or Kaggle into handoff/ (gitignored):
gen/, measure/sysinfo.py, the FramePack code (no weights), data/pilot5.json,
data/noedit5.json, the five reference images, and VERSION.json with the commits."""
import argparse, datetime, json, os, subprocess, zipfile

INCLUDE_DIRS = [("gen", lambda p: ".venv" not in p and "__pycache__" not in p),
                ("external/AlignVid/models/FramePack", lambda p: "hf_download" not in p and "__pycache__" not in p and not p.endswith(".7z"))]
INCLUDE_FILES = ["measure/sysinfo.py", "data/pilot5.json", "data/noedit5.json", "protocol/pilot_protocol.md"]


def git(args, cwd=None):
    try:
        return subprocess.check_output(["git"] + args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="handoff/pilot_package.zip")
    ap.add_argument("--rootname", default="pilot_package")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    version = {"packaged": datetime.datetime.now().isoformat(timespec="seconds"), "repo_commit": git(["rev-parse", "--short", "HEAD"]),
               "alignvid_commit": git(["rev-parse", "HEAD"], cwd="external/AlignVid"),
               "protocol_frozen_commit": "ec4d9e4", "protocol_frozen_on": "2026-11-18"}
    pilot = json.load(open("data/pilot5.json", encoding="utf-8"))
    n = 0
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        def add(path, arc=None):
            nonlocal n; z.write(path, f"{a.rootname}/{arc or path}"); n += 1
        for d, keep in INCLUDE_DIRS:
            for dp, _, fs in os.walk(d):
                for f in fs:
                    p = os.path.join(dp, f)
                    if keep(p):
                        add(p)
        for f in INCLUDE_FILES:
            add(f)
        for r in pilot:
            add(os.path.join("data/OmitI2V", r["image-path"]))
        z.writestr(f"{a.rootname}/VERSION.json", json.dumps(version, indent=1))
        z.writestr(f"{a.rootname}/README.txt", "Double-click gen\\round1.bat. Details in gen\\README_laptop.md.\n")
    print(f"wrote {a.out}: {n + 2} files, {round(os.path.getsize(a.out) / 2**20, 1)} MB"); print(json.dumps(version, indent=1))


if __name__ == "__main__":
    main()
