#!/usr/bin/env python3
"""Step B4: hardware / version report and run logging.  Standard library only;
torch and nvidia-smi are used when present.

  python measure/sysinfo.py                      # print JSON: GPU, VRAM, RAM, versions, repo + AlignVid commits
  python measure/sysinfo.py run --arm vanilla --id sample_240 -- python inference_f1_json.py ...
      runs the command, polls nvidia-smi for peak GPU memory used, records the
      child's peak RSS, wall time, exit code, stdout/stderr tails, and writes
      logs/<arm>_<id>.json together with the sysinfo block.

In-process use:
  from sysinfo import Run
  with Run("logs/vanilla_sample_240.json", arm="vanilla", id="sample_240") as r: ...
  (adds torch.cuda.max_memory_allocated when torch+CUDA are available)."""
import argparse, datetime, json, os, platform, re, shutil, subprocess, sys, threading, time
try:
    import resource  # not available on Windows
except ImportError:
    resource = None
try:
    import psutil
except ImportError:
    psutil = None

LIBS = ("torch", "torchvision", "diffusers", "transformers", "accelerate", "xformers", "flash_attn", "sageattention",
        "cv2", "numpy", "PIL", "imageio", "safetensors", "huggingface_hub", "psutil")


def sh(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL, text=True, timeout=20).strip()
    except Exception:
        return None


def nvidia_smi(query):
    if not shutil.which("nvidia-smi"):
        return None
    out = sh(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    return out.splitlines() if out else None


def gpu_info():
    info = {"nvidia_smi": None, "torch_cuda": None}
    rows = nvidia_smi("name,memory.total,driver_version")
    if rows:
        info["nvidia_smi"] = [dict(zip(("name", "vram_total_mb", "driver"), [v.strip() for v in r.split(",")])) for r in rows]
    try:
        import torch
        info["torch_cuda"] = {"available": torch.cuda.is_available(), "version": torch.version.cuda,
                              "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                              "total_mem_gb": [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)
                                               for i in range(torch.cuda.device_count())]}
    except Exception as e:
        info["torch_cuda"] = {"error": repr(e)}
    return info


def ram_gb():
    try:
        kb = int(re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read()).group(1))
        return round(kb / 2**20, 2)
    except Exception:
        return None


def versions():
    out = {"python": platform.python_version()}
    for n in LIBS:
        try:
            out[n] = getattr(__import__(n), "__version__", "present")
        except Exception:
            pass
    return out


def sysinfo(alignvid_dir="external/AlignVid"):
    return {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "host": platform.node(),
            "platform": platform.platform(), "cpu": platform.processor() or sh(["uname", "-p"]),
            "ram_total_gb": ram_gb(), "gpu": gpu_info(), "versions": versions(),
            "repo_commit": sh(["git", "rev-parse", "--short", "HEAD"]),
            "alignvid_commit": sh(["git", "rev-parse", "HEAD"], cwd=alignvid_dir) if os.path.isdir(alignvid_dir) else None,
            "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "PYTORCH_CUDA_ALLOC_CONF") if os.environ.get(k)}}


class GpuPoller(threading.Thread):
    """Polls nvidia-smi memory.used every `interval` s; .peak_mb is the max seen (None without nvidia-smi)."""

    def __init__(self, interval=1.0):
        super().__init__(daemon=True); self.interval = interval; self.peak_mb = None; self._stop = threading.Event()
        self.enabled = shutil.which("nvidia-smi") is not None

    def run(self):
        while self.enabled and not self._stop.is_set():
            rows = nvidia_smi("memory.used")
            if rows:
                try:
                    used = max(float(r) for r in rows)
                    self.peak_mb = used if self.peak_mb is None else max(self.peak_mb, used)
                except ValueError:
                    pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


class RssPoller(threading.Thread):
    """Polls a child process tree's RSS with psutil every `interval` s; .peak_gb is the max seen."""

    def __init__(self, pid, interval=1.0):
        super().__init__(daemon=True); self.pid = pid; self.interval = interval; self.peak_gb = None; self._stop = threading.Event()

    def run(self):
        if psutil is None:
            return
        while not self._stop.is_set():
            try:
                p = psutil.Process(self.pid)
                rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
                g = rss / 2**30; self.peak_gb = g if self.peak_gb is None else max(self.peak_gb, g)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


def self_peak_rss_gb():
    if resource is not None:
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)
    if psutil is not None:
        mi = psutil.Process().memory_info(); return round(getattr(mi, "peak_wset", mi.rss) / 2**30, 3)
    return None


class Run:
    """Context manager for an in-process block: wall time, peak RSS, torch CUDA peak, writes JSON on exit."""

    def __init__(self, log_path, **meta):
        self.log_path, self.meta = log_path, meta

    def __enter__(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self.poller = GpuPoller(); self.poller.start()
        self.t0 = time.time(); self.start = datetime.datetime.now().isoformat(timespec="seconds")
        return self

    def __exit__(self, et, ev, tb):
        self.poller.stop()
        rec = {**self.meta, "start": self.start, "end": datetime.datetime.now().isoformat(timespec="seconds"),
               "wall_seconds": round(time.time() - self.t0, 2), "ok": et is None, "error": None if et is None else repr(ev),
               "peak_rss_gb": self_peak_rss_gb(),
               "peak_gpu_used_mb_nvidia_smi": self.poller.peak_mb, "torch_cuda_max_allocated_gb": None, "sysinfo": sysinfo()}
        try:
            import torch
            if torch.cuda.is_available():
                rec["torch_cuda_max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        except Exception:
            pass
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        json.dump(rec, open(self.log_path, "w"), indent=1)
        return False


def run_command(cmd, log_path, cwd=None, tail=60, echo=True, **meta):
    """Run cmd as a child, stream its output, record wall time, exit code, child peak RSS, nvidia-smi peak."""
    poller = GpuPoller(); poller.start()
    t0 = time.time(); start = datetime.datetime.now().isoformat(timespec="seconds")
    lines = []
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         encoding="utf-8", errors="replace")
    rss = RssPoller(p.pid); rss.start()
    for ln in p.stdout:
        if echo:
            sys.stdout.write(ln); sys.stdout.flush()
        lines.append(ln.rstrip("\n"))
        if len(lines) > 5000:
            lines = lines[-tail:]
    p.wait(); poller.stop(); rss.stop()
    if rss.peak_gb is not None:
        child_peak = round(rss.peak_gb, 3); child_peak_src = "psutil poll 1 s"
    elif resource is not None:
        child_peak = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 3); child_peak_src = "ru_maxrss children (cumulative)"
    else:
        child_peak = None; child_peak_src = None
    rec = {**meta, "command": cmd, "cwd": cwd, "start": start, "end": datetime.datetime.now().isoformat(timespec="seconds"),
           "wall_seconds": round(time.time() - t0, 2), "exit_code": p.returncode, "ok": p.returncode == 0,
           "child_peak_rss_gb": child_peak, "child_peak_rss_source": child_peak_src,
           "peak_gpu_used_mb_nvidia_smi": poller.peak_mb, "output_tail": lines[-tail:], "sysinfo": sysinfo()}
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    json.dump(rec, open(log_path, "w"), indent=1)
    print(f"[sysinfo] {log_path}: exit {p.returncode}, {rec['wall_seconds']} s, child peak RSS {rec['child_peak_rss_gb']} GB, "
          f"GPU peak {poller.peak_mb} MB")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode")
    r = sub.add_parser("run", help="run a command and log it")
    r.add_argument("--arm", required=True); r.add_argument("--id", required=True)
    r.add_argument("--log-dir", default="logs"); r.add_argument("--cwd", default=None)
    r.add_argument("cmd", nargs=argparse.REMAINDER, help="command after --")
    a = ap.parse_args()
    if a.mode == "run":
        cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
        if not cmd:
            ap.error("give the command after --")
        rec = run_command(cmd, os.path.join(a.log_dir, f"{a.arm}_{a.id}.json"), cwd=a.cwd, arm=a.arm, id=a.id)
        sys.exit(rec["exit_code"])
    print(json.dumps(sysinfo(), indent=1))


if __name__ == "__main__":
    main()
