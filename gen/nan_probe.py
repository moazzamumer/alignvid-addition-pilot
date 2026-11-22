#!/usr/bin/env python3
"""Diagnostic: find the first module of the FramePack transformer whose output is non-finite.

Runs the unmodified generator (inference_f1_json.py, cwd = FramePack folder) with a forward hook on every
transformer submodule. On the first NaN/Inf output it prints the module name, its input/parameter
magnitudes and the max |x| of the 25 module outputs before it, then exits with code 3.
If the first full transformer call is finite it reports the output magnitude and exits 0
(set NAN_PROBE_EXIT_AFTER_FIRST=0 to let the generation continue).
Used once at C3 (2026-11-22) to locate the fp16 overflow on a Tesla T4. No model code is changed.

  cd external/AlignVid/models/FramePack
  PILOT_TRANSFORMER_DTYPE=fp16 python <repo>/gen/nan_probe.py --data_dir ... --prompt_file ... --output_dir ... [generator flags]
"""
import os, sys, runpy
import torch

RECENT = []


def _maxabs(t):
    return float(t.detach().abs().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).max()) if t.numel() else 0.0


def install(model):
    names = {m: n for n, m in model.named_modules()}

    def hook(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) and out else out
        if not torch.is_tensor(t) or not t.is_floating_point():
            return
        finite = bool(torch.isfinite(t).all())
        RECENT.append((names[mod], type(mod).__name__, str(t.dtype), round(_maxabs(t), 1)))
        del RECENT[:-25]
        if finite:
            return
        print("\n=== NAN PROBE: first non-finite module output ===")
        print("module :", names[mod], type(mod).__name__)
        print("output :", str(t.dtype), tuple(t.shape), "nan", int(torch.isnan(t).sum()), "inf", int(torch.isinf(t).sum()), "of", t.numel())
        ins = []
        for x in inp:
            if torch.is_tensor(x):
                ins.append((tuple(x.shape), str(x.dtype), "finite" if bool(torch.isfinite(x).all()) else "NONFINITE", round(_maxabs(x), 1)))
            else:
                ins.append(type(x).__name__)
        print("inputs :", ins)
        print("params :", [(n, str(p.dtype), round(float(p.detach().float().abs().max()), 3)) for n, p in mod.named_parameters(recurse=False)])
        print("--- previous 25 module outputs (name, class, dtype, max|x|) ---")
        for r in RECENT[:-1]:
            print("  ", r)
        sys.stdout.flush()
        os._exit(3)

    for m in names:
        m.register_forward_hook(hook)

    def done(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else getattr(out, "sample", out)
        print(f"\n=== NAN PROBE: first full transformer call finite, output {str(t.dtype)} {tuple(t.shape)} max|x| {round(_maxabs(t), 3)} ===")
        sys.stdout.flush()
        if os.environ.get("NAN_PROBE_EXIT_AFTER_FIRST", "1") != "0":
            os._exit(0)

    model.register_forward_hook(done)
    print(f"NAN PROBE installed on {len(names)} modules")


def main():
    sys.path.insert(0, os.getcwd())
    import diffusers_helper.models.hunyuan_video_packed as hvp
    cls = hvp.HunyuanVideoTransformer3DModelPacked
    orig = cls.from_pretrained.__func__

    def from_pretrained(klass, *a, **k):
        m = orig(klass, *a, **k)
        install(m)
        return m

    cls.from_pretrained = classmethod(from_pretrained)
    sys.argv = ["inference_f1_json.py"] + sys.argv[1:]
    runpy.run_path("inference_f1_json.py", run_name="__main__")


if __name__ == "__main__":
    main()
