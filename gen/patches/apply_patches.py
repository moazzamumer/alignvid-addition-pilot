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
    {"id": "sdpa-query-chunks", "file": f"{FP}/diffusers_helper/models/hunyuan_video_packed.py",
     "old": "        x = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),\n                                                             v.transpose(1, 2)).transpose(1, 2)\n        return x\n",
     "new": "        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)\n"
            "        # Pilot patch: on GPUs without a fused bf16 attention kernel (compute capability < 8.0) the\n"
            "        # math fallback materialises the full score matrix; run it in query chunks instead (exact).\n"
            "        _chunk = int(os.environ.get('ATTN_Q_CHUNK', '512' if qt.is_cuda and torch.cuda.get_device_capability(qt.device)[0] < 8 else '0'))\n"
            "        if _chunk <= 0 or qt.shape[2] <= _chunk:\n"
            "            x = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt)\n"
            "        else:\n"
            "            x = torch.cat([torch.nn.functional.scaled_dot_product_attention(qt[:, :, i:i + _chunk], kt, vt)\n"
            "                           for i in range(0, qt.shape[2], _chunk)], dim=2)\n"
            "        return x.transpose(1, 2)\n",
     "why": "Tesla T4 (sm_75) has no flash/mem-efficient bf16 SDPA kernel; the math fallback tried to allocate 30 GiB for the score matrix (C3 OOM, 2026-11-22). Query chunking computes the same softmax(QK^T)V per query row; only the memory layout changes. Inactive on sm_80+ unless ATTN_Q_CHUNK is set."},
    {"id": "transformer-dtype-env", "file": f"{FP}/inference_f1_json.py",
     "edits": [
         ("    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(\"lllyasviel/FramePack_F1_I2V_HY_20250503\", torch_dtype=torch.bfloat16).cpu()\n",
          "    # Pilot patch: transformer compute dtype from PILOT_TRANSFORMER_DTYPE (default bf16 = upstream; fp16 for GPUs\n"
          "    # without native bf16). The other models are fp16 upstream already; the final projection stays fp32.\n"
          "    _TRANSFORMER_DTYPE = torch.float16 if os.environ.get('PILOT_TRANSFORMER_DTYPE', 'bf16').lower() in ('fp16', 'float16') else torch.bfloat16\n"
          "    print(f'Pilot: transformer dtype {_TRANSFORMER_DTYPE}')\n"
          "    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(\"lllyasviel/FramePack_F1_I2V_HY_20250503\", torch_dtype=_TRANSFORMER_DTYPE).cpu()\n"),
         ("    transformer.to(dtype=torch.bfloat16)\n",
          "    transformer.to(dtype=_TRANSFORMER_DTYPE)\n"),
         ("                device=gpu,\n                dtype=torch.bfloat16,\n                image_embeddings=image_embeddings,\n",
          "                device=gpu,\n                dtype=transformer.dtype,\n                image_embeddings=image_embeddings,\n"),
         ("                device=gpu,\n                dtype=torch.bfloat16,\n                image_embeddings=image_encoder_last_hidden_state,\n",
          "                device=gpu,\n                dtype=transformer.dtype,\n                image_embeddings=image_encoder_last_hidden_state,\n"),
     ],
     "why": "Tesla T4 (sm_75) emulates bf16: the bf16 dry run took 8358 s for a 2 s clip (C3, 2026-11-22). fp16 uses the T4 tensor cores. Off unless PILOT_TRANSFORMER_DTYPE=fp16; the value used is recorded in each attempt log."},
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def apply(root):
    recs = []
    for pt in PATCHES:
        path = os.path.join(root, pt["file"])
        rec = {"id": pt["id"], "file": pt["file"], "why": pt["why"], "sha256_before": sha(path), "applied": False, "already": False}
        s = open(path, encoding="utf-8").read()
        edits = pt.get("edits") or [(pt["old"], pt["new"])]
        done = missing = 0
        for old, new in edits:
            if new in s:
                done += 1
            elif old in s:
                s = s.replace(old, new, 1); rec["applied"] = True
            else:
                missing += 1
        if missing:
            rec["error"] = f"pattern not found ({missing} of {len(edits)} edits)"
        elif done == len(edits):
            rec["already"] = True
        if rec["applied"]:
            open(path, "w", encoding="utf-8", newline="").write(s)
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
