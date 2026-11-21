#!/usr/bin/env python3
"""Phase C orchestration (C1 env check, C3 dry run, C4 generation).  Runs on the
generation machine from the folder that holds gen/, measure/, data/ and
external/AlignVid.  Every FramePack call is one child process per video wrapped by
measure/sysinfo.run_command, so each video has its own wall time, peak GPU memory
and peak RAM in logs/<arm>_<id>__len<L>_gm<G>.json (one file per attempt, failures kept)
and a slim sidecar outputs/<arm>/<id>.json.

Stages (all resumable; existing mp4 files are skipped):
  env      logs/env_<host>.json + .txt: GPU, driver, RAM, versions, bf16 matmul test
  patch    apply gen/patches (distributed backend fallback), record sha256 before/after
  weights  snapshot_download of the three model repos into FramePack/hf_download (resumable)
  dryrun   vanilla sample_240 at --length 2.0 then 1.0 with --gpu_mem 6 (protocol C3)
  length   probe vanilla sample_240 at 6.0 s, else 3.0, else 2.0 (protocol §6); the
           successful probe IS the vanilla video for that sample
  arms     vanilla (1.0), alignvid (1.35), noedit (vanilla settings, §3.3 prompts)
  pack     results_<host>_<date>.zip with outputs/ and logs/

  python gen/run_pilot.py                 # everything
  python gen/run_pilot.py --stage env     # one stage
  python gen/run_pilot.py --skip-noedit
  python gen/run_pilot.py --dry           # local test: a fake generator replaces FramePack
  python gen/run_pilot.py --quiet --sync-dir /kaggle/working   # Kaggle"""
import argparse, datetime, hashlib, json, os, platform, shutil, subprocess, sys, time, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "measure"))
from sysinfo import sysinfo, run_command  # noqa: E402

FP_DIR = os.path.join(ROOT, "external", "AlignVid", "models", "FramePack")
DATA_DIR = os.path.join(ROOT, "data", "OmitI2V")
FIXED = ["--seed", "31337", "--steps", "25", "--cfg", "1.0", "--gs", "10.0", "--rs", "0.0", "--teacache", "True"]
ARMS = {"vanilla": ("1.0", "data/pilot5.json"), "alignvid": ("1.35", "data/pilot5.json"), "noedit": ("1.0", "data/noedit5.json")}
PROBE_ID = "sample_240"
DRYRUN_LADDER = [("2.0", "6"), ("1.0", "6")]  # C3 on a 6 GB card: shorter clip if OOM; 6 GB preserved is already the maximum useful value
LENGTH_LADDER = ["6.0", "3.0", "2.0"]  # §6
HOST = os.environ.get("PILOT_HOST") or platform.node() or "machine"
QUIET = False
SYNC_DIR = None
LOGS = os.path.join(ROOT, "logs"); OUT = os.path.join(ROOT, "outputs"); STATE = os.path.join(LOGS, "run_state.json")


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(LOGS, exist_ok=True)
    open(os.path.join(LOGS, "run_pilot.log"), "a", encoding="utf-8").write(line + "\n")


def state(**kw):
    s = json.load(open(STATE)) if os.path.exists(STATE) else {}
    s.update(kw); json.dump(s, open(STATE, "w"), indent=1); return s


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def version_info():
    v = os.path.join(ROOT, "VERSION.json")
    return json.load(open(v)) if os.path.exists(v) else {}


# ---------------- stage: env ----------------
def stage_env():
    info = sysinfo(os.path.join(ROOT, "external", "AlignVid"))
    info["version_file"] = version_info()
    info["free_disk_gb"] = round(shutil.disk_usage(ROOT).free / 2**30, 1)
    test = {"bf16_matmul": None, "arch_list": None, "error": None}
    try:
        import torch
        test["arch_list"] = torch.cuda.get_arch_list()
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0); test["capability"] = f"sm_{cap[0]}{cap[1]}"
            a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            torch.cuda.synchronize(); t = time.time(); b = a @ a; torch.cuda.synchronize()
            test["bf16_matmul"] = {"ok": bool(torch.isfinite(b).all().item()), "seconds": round(time.time() - t, 4)}
            test["sdpa_bf16"] = bool(torch.isfinite(torch.nn.functional.scaled_dot_product_attention(
                a[None, None, :64], a[None, None, :64], a[None, None, :64])).all().item())
        else:
            test["error"] = "torch.cuda.is_available() is False"
    except Exception as e:
        test["error"] = repr(e)
    info["gpu_test"] = test
    os.makedirs(LOGS, exist_ok=True)
    json.dump(info, open(os.path.join(LOGS, f"env_{HOST}.json"), "w"), indent=1)
    txt = [f"env check {info['timestamp']} host={HOST}", f"platform: {info['platform']}", f"RAM: {info['ram_total_gb']} GB",
           f"free disk: {info['free_disk_gb']} GB", f"GPU: {json.dumps(info['gpu'])}", f"versions: {json.dumps(info['versions'])}",
           f"gpu_test: {json.dumps(test)}", f"alignvid_commit: {info['alignvid_commit'] or info['version_file'].get('alignvid_commit')}"]
    open(os.path.join(LOGS, f"env_{HOST}.txt"), "w").write("\n".join(txt) + "\n")
    for t in txt:
        log(t)
    ok = test["bf16_matmul"] is not None and test["bf16_matmul"]["ok"]
    if not ok:
        log("GPU test FAILED; stopping. Send the logs folder back.")
    return ok


# ---------------- stage: patch ----------------
def stage_patch():
    p = os.path.join(FP_DIR, "diffusers_helper", "parallel_mgr.py")
    old = "backend='nccl'"
    new = "backend=('nccl' if torch.distributed.is_nccl_available() else 'gloo')"
    rec = {"file": os.path.relpath(p, ROOT), "sha256_before": sha(p), "applied": False, "already": False, "original": old, "replacement": new}
    s = open(p, encoding="utf-8").read()
    if new in s:
        rec["already"] = True
    elif old in s:
        open(p, "w", encoding="utf-8").write(s.replace(old, new)); rec["applied"] = True
    else:
        rec["error"] = "pattern not found"
    rec["sha256_after"] = sha(p)
    json.dump(rec, open(os.path.join(LOGS, "patches.json"), "w"), indent=1)
    log(f"patch parallel_mgr.py: applied={rec['applied']} already={rec['already']} {rec.get('error', '')}")
    return "error" not in rec


# ---------------- stage: weights ----------------
WEIGHTS = [("lllyasviel/FramePack_F1_I2V_HY_20250503", None),
           ("hunyuanvideo-community/HunyuanVideo", ["text_encoder/*", "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*", "vae/*", "*.json"]),
           ("lllyasviel/flux_redux_bfl", ["feature_extractor/*", "image_encoder/*", "*.json"])]


def stage_weights():
    """Download the three model repos into FramePack's own hf_download cache (the inference
    script hard-codes HF_HOME there), skipping HunyuanVideo's unused 13B transformer.  Resumable."""
    hf_home = os.path.join(FP_DIR, "hf_download"); os.environ["HF_HOME"] = hf_home; os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.makedirs(hf_home, exist_ok=True)
    from huggingface_hub import snapshot_download
    rec = {"hf_home": hf_home, "repos": []}
    t_all = time.time()
    for repo, patterns in WEIGHTS:
        t = time.time(); log(f"weights: {repo} ...")
        try:
            path = snapshot_download(repo, allow_patterns=patterns, max_workers=4)
            size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)
            rec["repos"].append({"repo": repo, "ok": True, "seconds": round(time.time() - t, 1), "gb_on_disk": round(size / 2**30, 2)})
            log(f"weights: {repo} ok, {rec['repos'][-1]['gb_on_disk']} GB, {rec['repos'][-1]['seconds']} s")
        except Exception as e:
            rec["repos"].append({"repo": repo, "ok": False, "error": repr(e)}); log(f"weights: {repo} FAILED {e!r}")
    rec["total_seconds"] = round(time.time() - t_all, 1)
    json.dump(rec, open(os.path.join(LOGS, "weights.json"), "w"), indent=1)
    return all(r["ok"] for r in rec["repos"])


# ---------------- generation ----------------
def one_record_file(prompt_file, sid, out_dir):
    recs = [r for r in json.load(open(os.path.join(ROOT, prompt_file), encoding="utf-8")) if r["id"] == sid]
    p = os.path.join(out_dir, f"{sid}.prompt.json"); os.makedirs(out_dir, exist_ok=True)
    json.dump(recs, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False); return p


def sync(paths):
    """Copy finished files to SYNC_DIR (e.g. /kaggle/working) keeping the relative layout."""
    if not SYNC_DIR:
        return
    for p in paths:
        if os.path.exists(p):
            dst = os.path.join(SYNC_DIR, os.path.relpath(p, ROOT)); os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(p, dst)


def generate(arm, sid, length, gpu_mem, dry=False, out_dir=None):
    """One FramePack process for one video.  Returns the run record (ok True/False)."""
    control, prompt_file = ARMS[arm]
    out_dir = out_dir or os.path.join(OUT, arm)
    mp4 = os.path.join(out_dir, f"{sid}.mp4")
    if os.path.exists(mp4):
        log(f"{arm}/{sid}: exists, skipped"); return {"ok": True, "skipped": True}
    pf = one_record_file(prompt_file, sid, out_dir)
    if dry:
        cmd = [sys.executable, os.path.join(ROOT, "gen", "fake_generator.py"), "--output_dir", out_dir, "--prompt_file", pf,
               "--length", str(length), "--gpu_mem", str(gpu_mem), "--control_scale", control]
    else:
        cmd = [sys.executable, "inference_f1_json.py", "--data_dir", DATA_DIR, "--prompt_file", pf, "--output_dir", out_dir,
               "--length", str(length), "--gpu_mem", str(gpu_mem), "--control_scale", control] + FIXED
    env = {**os.environ, "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29555",
           "HF_HUB_DISABLE_XET": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "MPLBACKEND": "Agg"}
    os.environ.update(env)
    log(f"{arm}/{sid}: start length={length} gpu_mem={gpu_mem} control_scale={control}")
    log_path = os.path.join(LOGS, f"{arm}_{sid}__len{length}_gm{gpu_mem}.json")
    rec = run_command(cmd, log_path, cwd=FP_DIR, echo=not QUIET, arm=arm, id=sid, length=float(length),
                      gpu_mem=float(gpu_mem), control_scale=float(control), fixed_flags=FIXED, seed=31337, version_file=version_info())
    rec["ok"] = rec["ok"] and os.path.exists(mp4)
    if rec["ok"]:
        side = {k: rec[k] for k in ("arm", "id", "length", "gpu_mem", "control_scale", "fixed_flags", "seed", "start", "end",
                                     "wall_seconds", "child_peak_rss_gb", "peak_gpu_used_mb_nvidia_smi", "version_file")}
        side["versions"] = rec["sysinfo"]["versions"]; side["gpu"] = rec["sysinfo"]["gpu"]; side["host"] = HOST
        json.dump(side, open(os.path.join(out_dir, f"{sid}.json"), "w"), indent=1)
        log(f"{arm}/{sid}: done in {rec['wall_seconds']} s, GPU peak {rec['peak_gpu_used_mb_nvidia_smi']} MB, RSS {rec['child_peak_rss_gb']} GB")
        sync([mp4, os.path.join(out_dir, f"{sid}.json"), log_path, os.path.join(LOGS, "run_pilot.log"), STATE])
    else:
        log(f"{arm}/{sid}: FAILED exit {rec['exit_code']} after {rec['wall_seconds']} s; tail: {' | '.join(rec['output_tail'][-3:])}")
        sync([log_path, os.path.join(LOGS, "run_pilot.log")])
        for f in os.listdir(out_dir):  # remove partial outputs for this id so a rerun is clean
            if f.startswith(sid) and f.endswith(".mp4"):
                os.remove(os.path.join(out_dir, f))
    return rec


def stage_dryrun(dry):
    st = state()
    if st.get("gpu_mem"):
        log(f"dryrun: already done, gpu_mem={st['gpu_mem']}"); return True
    for L, gm in DRYRUN_LADDER:
        rec = generate("vanilla", PROBE_ID, L, gm, dry, out_dir=os.path.join(OUT, "dryrun"))
        if rec["ok"]:
            state(gpu_mem=gm, dryrun={"length": float(L), "gpu_mem": gm, "wall_seconds": rec.get("wall_seconds")}); return True
    log("dryrun FAILED at every rung; stopping. Send the logs folder back.")
    state(dryrun_failed=True); return False


def stage_length(dry):
    st = state(); gm = st["gpu_mem"]
    if st.get("length"):
        log(f"length: already chosen {st['length']}"); return True
    for L in LENGTH_LADDER:
        rec = generate("vanilla", PROBE_ID, L, gm, dry)
        if rec["ok"]:
            state(length=L, length_probe={"length": L, "wall_seconds": rec.get("wall_seconds")}); log(f"length chosen: {L} s"); return True
    log("length probe FAILED at 6.0, 3.0 and 2.0; stopping."); state(length_failed=True); return False


def stage_arms(dry, skip_noedit, arms=None):
    st = state(); L, gm = st["length"], st["gpu_mem"]
    ids = [r["id"] for r in json.load(open(os.path.join(ROOT, "data", "pilot5.json"), encoding="utf-8"))]
    results = st.get("results", {})
    for arm in arms or (["vanilla", "alignvid"] + ([] if skip_noedit else ["noedit"])):
        for sid in ids:
            rec = generate(arm, sid, L, gm, dry)
            results[f"{arm}/{sid}"] = {"ok": rec["ok"], "wall_seconds": rec.get("wall_seconds"), "exit": rec.get("exit_code")}
            state(results=results)
    n_ok = sum(1 for v in results.values() if v["ok"]); log(f"arms: {n_ok}/{len(results)} videos ok")
    return True


def stage_pack():
    name = f"results_{HOST}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.zip"
    p = os.path.join(ROOT, name)
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        for d in (OUT, LOGS):
            for dp, _, fs in os.walk(d):
                for f in fs:
                    z.write(os.path.join(dp, f), os.path.relpath(os.path.join(dp, f), ROOT))
    log(f"packed {name} ({round(os.path.getsize(p) / 2**20, 1)} MB). Upload this file.")
    sync([p] + [os.path.join(dp, f) for dp, _, fs in os.walk(LOGS) for f in fs])
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["env", "patch", "weights", "dryrun", "length", "arms", "pack"], help="run one stage only")
    ap.add_argument("--skip-noedit", action="store_true")
    ap.add_argument("--dry", action="store_true", help="local test with gen/fake_generator.py")
    ap.add_argument("--quiet", action="store_true", help="do not echo FramePack's own output (it is kept in the per-attempt log)")
    ap.add_argument("--sync-dir", help="copy finished videos, sidecars and logs here as they appear (Kaggle: /kaggle/working)")
    ap.add_argument("--arms", nargs="*", choices=list(ARMS), help="restrict the arms stage, e.g. --arms vanilla")
    a = ap.parse_args()
    global QUIET, SYNC_DIR
    QUIET, SYNC_DIR = a.quiet, a.sync_dir
    os.makedirs(LOGS, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    log(f"run_pilot start root={ROOT} dry={a.dry} version={version_info()}")
    stages = [("env", lambda: a.dry or stage_env()), ("patch", stage_patch), ("weights", lambda: a.dry or stage_weights()),
              ("dryrun", lambda: stage_dryrun(a.dry)),
              ("length", lambda: stage_length(a.dry)), ("arms", lambda: stage_arms(a.dry, a.skip_noedit, a.arms)), ("pack", stage_pack)]
    for name, fn in stages:
        if a.stage and a.stage != name:
            continue
        log(f"--- stage {name} ---")
        if not fn():
            if name in ("env", "weights", "dryrun", "length"):
                stage_pack()
            log(f"stopped at stage {name}"); sys.exit(1)
    log("all stages finished")


if __name__ == "__main__":
    main()
