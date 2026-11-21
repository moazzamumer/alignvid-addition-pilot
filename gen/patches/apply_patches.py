#!/usr/bin/env python3
"""Apply the recorded source patches to the FramePack copy under external/AlignVid.
Idempotent; writes logs/patches.json with sha256 before/after.  See gen/patches/README.md.
  python gen/patches/apply_patches.py [--root <folder that holds external/ and logs/>]"""
import argparse, hashlib, json, os

FP = "external/AlignVid/models/FramePack"
PATCHES = [
    {"id": "dist-backend", "file": f"{FP}/diffusers_helper/parallel_mgr.py",
     "old": "backend='nccl'",
     "new": "backend=('nccl' if torch.distributed.is_nccl_available() else 'gloo')",
     "why": "NCCL does not exist on Windows; one-process group init before any model is loaded."},
    {"id": "lowvram-text-encode-1", "file": f"{FP}/inference_f1_json.py",
     "old": "        if not high_vram:\n            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)\n\n        # Encode prompts\n",
     "new": "        if not high_vram:\n            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)\n            fake_diffusers_current_device(text_encoder, gpu)\n            load_model_as_complete(text_encoder_2, target_device=gpu)\n\n        # Encode prompts\n",
     "why": "Low-VRAM branch (free VRAM <= 60 GB) lacked the two upstream FramePack lines that place the text encoders before encoding; without them input ids go to CPU while the swapped embedding weights are on the GPU (device mismatch at C3 on Kaggle T4)."},
    {"id": "lowvram-text-encode-2", "file": f"{FP}/inference_f1_json.py",
     "old": "            llama_vec, clip_l_pooler = encode_prompt_conds(\n                active_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2\n            )\n",
     "new": "            if not high_vram:\n                fake_diffusers_current_device(text_encoder, gpu)\n                load_model_as_complete(text_encoder_2, target_device=gpu)\n            llama_vec, clip_l_pooler = encode_prompt_conds(\n                active_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2\n            )\n",
     "why": "Same fix for the per-section prompt encode inside the generation loop."},
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def apply(root):
    recs = []
    for pt in PATCHES:
        path = os.path.join(root, pt["file"])
        rec = {"id": pt["id"], "file": pt["file"], "why": pt["why"], "sha256_before": sha(path), "applied": False, "already": False}
        s = open(path, encoding="utf-8").read()
        if pt["new"] in s:
            rec["already"] = True
        elif pt["old"] in s:
            open(path, "w", encoding="utf-8", newline="").write(s.replace(pt["old"], pt["new"], 1)); rec["applied"] = True
        else:
            rec["error"] = "pattern not found"
        rec["sha256_after"] = sha(path); recs.append(rec)
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    json.dump(recs, open(os.path.join(root, "logs", "patches.json"), "w"), indent=1)
    return recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    a = ap.parse_args()
    for r in apply(a.root):
        print(f"{r['id']}: applied={r['applied']} already={r['already']} {r.get('error', '')}")
