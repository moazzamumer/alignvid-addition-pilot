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
| OmitI2V snapshot | HF `AIPeanutman/OmitI2V`, downloaded 2026-11-18 via `huggingface_hub` 2.4.0 snapshot_download (370 files, 273 MB, 367 records), `meta.json` sha256 `abc7099ba16ccb9f48d24b259f2343c48dc447f9907c511008df28a18723937d` |
| Backbone | FramePack-F1 (`lllyasviel/FramePack_F1_I2V_HY_20250503`) |
| DyMoS | not used; repo github.com/shoxedqBbB/DyMoS (noted: includes Wan2.1 code as of 2026-11) |

## 3. Data

Addition subset count from `meta.json`: 129 (sub-categories: object 73, appearance 23, element 33). Index of the 73 addition/object samples: `data/addition_object_index.csv` (built by `tools/inspect_meta.py` on 2026-11-18). Domains within addition/object: human 20, animal 16, object 12, animation 10, plant 6, vehicle 4, architecture 3, building 2. Image source: real 62, animation 10, generated 1. All 73 reference images present on disk.

Metadata quirks noted before selection: `sample_121` (human) has prompt "The woman gradually disappeared as she crouched down" but key "duck egg appears" — prompt/key mismatch, excluded from consideration. `sample_199` has one question with an empty category string. `sample_6` (object) describes drifting dust, not a placeable object.

### 3.1 Selection rules (written before selecting)
1. `main-category == addition` and `sub-category == object`. (Note: in `meta.json` the field `type` is the image source — real / animation / generated — not the edit type.)
2. Prompt names one physical object with an explicit or scene-implied location.
3. Three straightforward cases: medium/large object, uncluttered target area.
4. Two difficult cases: one small object; one gradual ("grows"/"appears") or contact-with-subject prompt.
5. Exclude: object moving across the frame; object already present in the reference (checked by eye); near-duplicate scenes.
6. Operational definitions for rule 5 (added 2026-11-18, before selection): "moving across the frame" = the prompt's main verb describes translation into or across the frame, or entry from an edge (walks in, flies in, darts out, slides in, drifts across, enters from the left). Prompts whose verb is "appears", "sits", "stands", "leans", "grows", "sprouts" at a named place are eligible. "Small object" = would occupy well under ~5 % of the frame at the named location. "Gradual" = prompt uses grows / sprouts / emerges / gradually.
7. Tie-breaks when more candidates than slots: prefer real photographs over animation or generated images; prefer prompts where the paper's own question set includes an `object_presence` or `spatial_relationship` question (so the original protocol can be compared later); prefer distinct domains across the five.

### 3.2 Chosen samples

Selected 2026-11-18 by applying rules 1–7 to `data/addition_object_index.csv`, before any video was generated. Shortlist of 13 candidates reviewed on a contact sheet (`tools/contact_sheet.py`); source presence confirmed absent by eye by the author (M. Umer Gondal) for all five on 2026-11-18.

| # | id | prompt | key | difficulty | source presence (by eye) | justification |
|---|---|---|---|---|---|---|
| 1 | sample_240 | An ornate, antique world globe on a wooden stand, reminiscent of the Victorian era, appears next to the sofa. | antique world globe appears | easy | no | Indoor, static scene; clear empty floor beside the sofa; medium object; domain object; real photo. |
| 2 | sample_188 | A person sits down in the empty chair at the table, picks up the book, and begins to read. | person sits down; begins to read | easy (contact) | no | Chair, table and book visible and unoccupied; fixed location; object must touch an existing subject; domain human; real photo. |
| 3 | sample_208 | A majestic stag with large antlers appears at the edge of the frozen lake, its breath visible in the cold air. | majestic stag appears; breath visible | easy | no | Medium-large animal in a static landscape; "edge of the lake" is ambiguous between near and far bank, so this is the designated multiple-valid-regions case; domain animal; real photo. Note: the lake in the reference is not frozen (autumn); recorded as a prompt/image mismatch to watch. |
| 4 | sample_245 | A large, jewel-encrusted Faberg egg appears in the shopping cart. | Faberg egg appears; in shopping cart | hard (small) | no | Small object relative to the frame, inside a clean, fully visible container, so the intended region is one unambiguous box; a moderate small-object case; domain object; real photo. Chosen over sample_237 to avoid two lakeside scenes (208 already is one) and to keep five visually distinct scene types. Note: the prompt spells "Faberg" (dataset typo for Fabergé); kept verbatim. |
| 5 | sample_116 | A young tree sprouts at the front of the house and grows quickly until it stands full-height in front of the facade. | tree grows; in front of house | hard (gradual) | no | Explicit location on bare ground in front of the facade; "sprouts and grows" tests the expected-appearance window; large final object; domain architecture; real photo. |

Named alternates, in case a chosen sample proves unusable (to be invoked only via §9 Deviations): easy → `sample_236` (red barn on the hill crest); small → `sample_237` (small Celtic cairn on the far bank; a harsher small-object case, very small and with near/far-bank ambiguity); gradual → `sample_102` (cactus sprouts next to the mushroom; gradual and contact-adjacent at once).

Shortlisted and rejected after viewing the reference image: `sample_228` (trees already in front of the building), `sample_131` (middle egg already has sprouts; animation image), `sample_239` (cable car "behind" the tram would be occluded), `sample_243` (pumpkins fill the frame, so the intended region is the whole frame), `sample_230` (crowded museum, non-static scene; fallback only).

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
