"""JSON sidecar written next to every measurement/generation output: arguments,
timing, peak memory, git hash and library versions.  Shared by all measure/ scripts."""
import datetime, json, os, platform, resource, subprocess, sys


def git_hash(path=None):
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=path or os.getcwd(),
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def versions(names=("cv2", "numpy", "PIL", "torch", "transformers", "torchvision", "diffusers")):
    out = {"python": platform.python_version()}
    for n in names:
        try:
            m = __import__(n)
            out[n] = getattr(m, "__version__", None)
        except Exception:
            pass
    return out


def peak_rss_mb():
    # ru_maxrss is KB on Linux
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


class Sidecar:
    """with Sidecar(path, args) as sc: ...; sc.add(key=value) ... -> writes JSON on exit."""

    def __init__(self, path, args=None, **extra):
        self.path = path
        self.data = {"script": os.path.basename(sys.argv[0]), "args": vars(args) if hasattr(args, "__dict__") else args,
                     "git_hash": git_hash(), "versions": versions(), "host": platform.node(), **extra}

    def __enter__(self):
        self.t0 = datetime.datetime.now()
        self.data["start"] = self.t0.isoformat(timespec="seconds")
        return self

    def add(self, **kw):
        self.data.update(kw)

    def __exit__(self, et, ev, tb):
        t1 = datetime.datetime.now()
        self.data.update(end=t1.isoformat(timespec="seconds"), wall_seconds=round((t1 - self.t0).total_seconds(), 2),
                         peak_rss_mb=peak_rss_mb(), ok=et is None, error=None if et is None else repr(ev))
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        json.dump(self.data, open(self.path, "w"), indent=1)
        return False
