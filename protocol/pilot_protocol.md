# Pilot protocol (pre-registration)

**Status:** DRAFT → mark FROZEN once complete
**Frozen on:** YYYY-MM-DD (Asia/Karachi)
**Frozen at commit:** `_______` (fill after the freeze commit; amend once)
**Author:** Moazzam Umer Gondal
**Second annotator:** ______ (name/role)

Everything below is fixed before any video is generated. Later changes go only under §9 Deviations, dated.

---

## 1. Question

Does the pipeline (data → generation → frame extraction → human labels → detector check → panel) run end to end on accessible hardware, and where do labels, tools or compute break down? n = 5 samples, 1 seed, 2 arms. No performance difference will be estimated.

## 2. Versions

| Item | Value |
|---|---|
| AlignVid repo commit | `9f3e48d76e19fa44e7ca93f0466ca92fb24775c2` (github.com/LAW1223/AlignVid, single commit "Initial release of AlignVid", 2026-05-29), added as submodule `external/AlignVid` on 2026-11-18 |
| AlignVid paper version | arXiv 2512.01334 v2 (2026-05-28) |
| OmitI2V snapshot | HF `AIPeanutman/OmitI2V`, downloaded YYYY-MM-DD, `meta.json` sha256 `_______` |
| Backbone | FramePack-F1 (`lllyasviel/FramePack_F1_I2V_HY_20250503`) |
| DyMoS | not used; repo github.com/shoxedqBbB/DyMoS (noted: includes Wan2.1 code as of 2026-11) |

## 3. Data

Addition subset count from `meta.json`: ___ (sub-categories: object ___, appearance ___, element ___).

### 3.1 Selection rules (written before selecting)
1. `type == addition` and `sub-category == object`.
2. Prompt names one physical object with an explicit or scene-implied location.
3. Three straightforward cases: medium/large object, uncluttered target area.
4. Two difficult cases: one small object; one gradual ("grows"/"appears") or contact-with-subject prompt.
5. Exclude: object moving across the frame; object already present in the reference (checked by eye); near-duplicate scenes.

### 3.2 Chosen samples

| # | id | prompt | key | difficulty | source presence (by eye) | justification |
|---|---|---|---|---|---|---|
| 1 | | | | easy | no | |
| 2 | | | | easy | no | |
| 3 | | | | easy | no | |
| 4 | | | | hard (small) | no | |
| 5 | | | | hard (gradual/contact) | no | |

### 3.3 No-edit control prompts (optional arm; written now)

| id | motion-neutral caption (no new object) |
|---|---|
| | |

## 4. Intended regions

### 4.1 Rules
- Box(es) in normalised `[x0,y0,x1,y1]` on the original image; multiple boxes allowed when the prompt permits; `free_placement` tag if no location can be inferred (no box).
- Drawn independently by two annotators from this text, then discussed; final = agreed box. IoU and the **type** of disagreement recorded.
- Mapping to generated frames: FramePack center-crops/resizes to a ~640-px bucket; the region is re-projected onto the first generated frame using the recorded crop.
- Placement rule for a detected/judged object box: **in** if ≥ 50 % of its area lies inside the (union of) region(s); **partial** if 10–50 %; **out** otherwise.

### 4.2 Boxes

| id | annotator A | annotator B | IoU | disagreement type | final |
|---|---|---|---|---|---|
| | | | | | |

Previews: `data/regions/<id>_box.jpg`.

## 5. Expected-appearance window

| id | window | reason |
|---|---|---|
| | visible by ≤ 25 % of frames and persisting to end | instantaneous prompt |
| | visible by ≤ 50 % and persisting | "grows" |

## 6. Generation settings (identical across arms except γ)

FramePack-F1 `inference_f1_json.py`: `--seed 31337 --length ___ --steps 25 --cfg 1.0 --gs 10.0 --rs 0.0 --gpu_mem ___ --teacache True`.
Arms: vanilla `--control_scale 1.0`; AlignVid `--control_scale 1.35`; (optional) no-edit = vanilla settings with §3.3 prompts.
Hardware intended: ______ (fill actual in the report).

## 7. Human labels

Rubric: `protocol/label_rubric.md` (frozen with this file). Labelling is blind to arm (random codes, map sealed in `labels/blind_map.csv`). Second labeller on all videos if available; report raw agreement per field and Cohen's κ for presence with n stated.

## 8. Detector settings (fixed on dev examples, not pilot samples)

Grounding DINO model: ______; box threshold ___; text threshold ___; frames per video 16; present if detected in ≥ k = ___ frames; dev images used: ______.
VLM: none / Qwen2.5-VL-7B stand-in (circle one). If used, per-question answers only; never pooled with the paper's protocol.

## 9. Deviations (dated)

| date | what changed | why |
|---|---|---|
| | | |
