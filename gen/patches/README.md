# Patches applied by gen/run_pilot.py before generation

| file | change | why |
|---|---|---|
| `models/FramePack/diffusers_helper/parallel_mgr.py` | `backend='nccl'` → `backend=('nccl' if torch.distributed.is_nccl_available() else 'gloo')` | NCCL does not exist on Windows; the script initialises a one-process group before loading any model. No model, sampling or attention code is touched. |

The original line is kept in `logs/patches.json` together with the file's sha256 before and after.
