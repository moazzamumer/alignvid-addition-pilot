# Source patches applied before generation (`gen/patches/apply_patches.py`)

Applied by `gen/run_pilot.py --stage patch`; each is recorded in `logs/patches.json` with the file's sha256 before and after. No model, sampling or attention code is touched.

| id | file | change | why |
|---|---|---|---|
| dist-backend | `diffusers_helper/parallel_mgr.py` | `backend='nccl'` → `'nccl' if available else 'gloo'` | NCCL does not exist on Windows; the script initialises a one-process group before loading any model. |
| lowvram-text-encode-1 | `inference_f1_json.py`, before the first prompt encode | add `fake_diffusers_current_device(text_encoder, gpu)` and `load_model_as_complete(text_encoder_2, target_device=gpu)` inside `if not high_vram:` | The low-VRAM branch (free VRAM ≤ 60 GB) is missing the two lines upstream FramePack has here. Without them `text_encoder.device` reports CPU while the dynamically swapped embedding weight is on the GPU: `RuntimeError: Expected all tensors to be on the same device` (found at C3 on a Kaggle T4, 2026-11-22). The authors' runs used ≥ 80 GB GPUs, where this branch is skipped. |
| lowvram-text-encode-2 | `inference_f1_json.py`, per-section prompt encode in the loop | same two lines | Same fix for the second call site. |
| sdpa-query-chunks | `diffusers_helper/models/hunyuan_video_packed.py`, SDPA fallback in `attn_varlen_func` | run `scaled_dot_product_attention` over query chunks of 512 when the GPU is older than Ampere (`ATTN_Q_CHUNK` overrides) | The T4 has no fused bf16 attention kernel, so the math fallback allocated the full score matrix (30 GiB, OOM at C3 on Kaggle, 2026-11-22). Chunking over queries is exact: each query row's softmax is independent. Inactive on sm_80+. |
