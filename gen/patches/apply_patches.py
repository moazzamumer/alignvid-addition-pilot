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
    {"id": "fp32-residual", "file": f"{FP}/diffusers_helper/models/hunyuan_video_packed.py",
     "edits": [
         ("        # 1. Input normalization\n        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)\n        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(\n            encoder_hidden_states, emb=temb)\n",
          "        # Pilot patch (fp32-residual): in fp16 the residual stream overflows (|x| > 29760 entering block 7 on\n"
          "        # sample_240, fp16 max 65504). Keep hidden/encoder states in fp32 between blocks; attention and MLPs\n"
          "        # still run in the compute dtype. No-op in bf16.\n"
          "        _c = temb.dtype\n"
          "        if _c == torch.float16:\n"
          "            hidden_states, encoder_hidden_states = hidden_states.float(), encoder_hidden_states.float()\n"
          "        # 1. Input normalization\n        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)\n        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(\n            encoder_hidden_states, emb=temb)\n"
          "        norm_hidden_states, norm_encoder_hidden_states = norm_hidden_states.to(_c), norm_encoder_hidden_states.to(_c)\n"),
         ("        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp\n        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp\n",
          "        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp\n        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp\n"
          "        norm_hidden_states, norm_encoder_hidden_states = norm_hidden_states.to(_c), norm_encoder_hidden_states.to(_c)\n"),
         ("        text_seq_length = encoder_hidden_states.shape[1]\n        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)\n\n        residual = hidden_states\n\n        # 1. Input normalization\n        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)\n        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))\n",
          "        text_seq_length = encoder_hidden_states.shape[1]\n        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)\n"
          "        # Pilot patch (fp32-residual): see HunyuanVideoTransformerBlock.forward.\n"
          "        _c = temb.dtype\n"
          "        if _c == torch.float16:\n"
          "            hidden_states = hidden_states.float()\n"
          "\n        residual = hidden_states\n\n        # 1. Input normalization\n        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)\n"
          "        norm_hidden_states = norm_hidden_states.to(_c)\n"
          "        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))\n"),
         ("        hidden_states = self.gradient_checkpointing_method(self.norm_out, hidden_states, temb)\n",
          "        if hidden_states.dtype == torch.float32 and temb.dtype == torch.float16:\n"
          "            # Pilot patch (fp32-residual): final AdaLayerNorm in fp32 too (same pattern as proj_out below).\n"
          "            self.norm_out.to(dtype=torch.float32)\n"
          "            temb = temb.float()\n"
          "        hidden_states = self.gradient_checkpointing_method(self.norm_out, hidden_states, temb)\n"),
         # gated products in the residual dtype: in fp16, ff_output (up to ~2e3) x gate (can exceed 30) overflowed
         # inside block 7 even with the fp32 residual (second NaN-probe run, 2026-11-22).
         ("        hidden_states = hidden_states + attn_output * gate_msa\n        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa\n",
          "        hidden_states = hidden_states + attn_output.to(hidden_states.dtype) * gate_msa\n        encoder_hidden_states = encoder_hidden_states + context_attn_output.to(encoder_hidden_states.dtype) * c_gate_msa\n"),
         ("        hidden_states = hidden_states + gate_mlp * ff_output\n        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output\n",
          "        hidden_states = hidden_states + gate_mlp * ff_output.to(hidden_states.dtype)\n        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output.to(encoder_hidden_states.dtype)\n"),
         ("        hidden_states = gate * self.proj_out(hidden_states)\n        hidden_states = hidden_states + residual\n",
          "        hidden_states = gate * self.proj_out(hidden_states).to(residual.dtype)\n        hidden_states = hidden_states + residual\n"),
     ],
     "why": "fp16 dry run on the T4 gave an all-black clip: the NaN probe (gen/nan_probe.py) showed the first Inf in transformer_blocks.7's residual add, with finite sub-layer outputs (<= 1940) but a residual stream already at 29760 (hidden) / 51328 (text). Large residual channels are a property of this model family. Residual kept in fp32 and the gated residual updates computed in fp32; attention and MLP matmuls stay fp16. Inactive in bf16."},
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
