# Pilot protocol (pre-registration)

**Status:** FROZEN
**Frozen on:** 2026-11-18 (Asia/Karachi)
**Frozen at commit:** `ec4d9e4`
**Author:** Moazzam Umer Gondal
**Second annotator:** MSc student (annotator B)

Fixed before any video is generated. Later changes only under §9 Deviations, dated.

---

## 1. Question

Does the pipeline (data → generation → frame extraction → human labels → detector check → panel) run end to end on accessible hardware, and where do labels, tools or compute break down? n = 5 samples, 1 seed, 2 arms. No performance difference will be estimated.

## 2. Versions

| Item | Value |
|---|---|
| AlignVid repo | github.com/LAW1223/AlignVid, commit `9f3e48d76e19fa44e7ca93f0466ca92fb24775c2` (2026-05-29, single commit), submodule `external/AlignVid` |
| AlignVid paper | arXiv 2512.01334 v2 (2026-05-28) |
| OmitI2V | HF `AIPeanutman/OmitI2V`, snapshot 2026-11-18 (367 records), `meta.json` sha256 `abc7099ba16ccb9f48d24b259f2343c48dc447f9907c511008df28a18723937d` |
| Backbone | FramePack-F1 (`lllyasviel/FramePack_F1_I2V_HY_20250503`) |
| Detector | Grounding DINO tiny (`IDEA-Research/grounding-dino-tiny`), `model.safetensors` sha256 prefix `1a2412ef99bd74bc` |
| Measurement env | Python 3.11.2, torch 2.15.0+cpu, transformers 5.23.0, opencv 5.1.0, numpy 2.4.6, pillow 13.0.0, huggingface_hub 2.4.0 |
| Generation env | recorded in `logs/env_<machine>.txt` at step C1 |
| DyMoS | not used; github.com/shoxedqBbB/DyMoS (includes Wan2.1 code as of 2026-11) |

## 3. Data

Addition subset in `meta.json`: 129 (object 73, appearance 23, element 33). Index of the 73 addition/object samples: `data/addition_object_index.csv` (`tools/inspect_meta.py`). Domains: human 20, animal 16, object 12, animation 10, plant 6, vehicle 4, architecture 3, building 2; real photographs 62, animation 10, generated 1.

Excluded before selection for metadata problems: `sample_121` (prompt/key mismatch), `sample_6` (drifting dust, no placeable object).

### 3.1 Selection rules (written before selecting)
1. `main-category == addition` and `sub-category == object` (`type` in `meta.json` is the image source, not the edit type).
2. Prompt names one physical object with an explicit or scene-implied location.
3. Three straightforward cases: medium/large object, uncluttered target area.
4. Two difficult cases: one small object; one gradual ("grows"/"appears") or contact-with-subject prompt.
5. Exclude: object moving across the frame; object already present in the reference (checked by eye); near-duplicate scenes.
6. Definitions: "moving across the frame" = the main verb describes translation into or across the frame or entry from an edge (walks in, flies in, darts out, slides in, drifts across); "appears / sits / stands / leans / grows / sprouts at a named place" is eligible. "Small" = under ~5 % of the frame at the named location. "Gradual" = grows / sprouts / emerges / gradually.
7. Tie-breaks: real photographs over animation/generated; prompts whose OmitI2V questions include `object_presence` or `spatial_relationship`; distinct domains across the five.

### 3.2 Chosen samples

Selected 2026-11-18 by rules 1–7 from a 13-candidate shortlist (`tools/contact_sheet.py`). Source presence checked by eye on the reference image.

| # | id | prompt | key | difficulty | source presence (by eye) | justification |
|---|---|---|---|---|---|---|
| 1 | sample_240 | An ornate, antique world globe on a wooden stand, reminiscent of the Victorian era, appears next to the sofa. | antique world globe appears | easy | no | Indoor static scene, clear floor beside the sofa, medium object. |
| 2 | sample_188 | A person sits down in the empty chair at the table, picks up the book, and begins to read. | person sits down; begins to read | easy (contact) | no | Empty chair and book visible; fixed location; object must touch an existing subject. |
| 3 | sample_208 | A majestic stag with large antlers appears at the edge of the frozen lake, its breath visible in the cold air. | majestic stag appears; breath visible | easy | no | Medium-large animal, static landscape. Prompt says "frozen"; the reference lake is open autumn water (dataset prompt/image mismatch, noted). |
| 4 | sample_245 | A large, jewel-encrusted Faberg egg appears in the shopping cart. | Faberg egg appears; in shopping cart | hard (small) | no | Small object in a fully visible container; one unambiguous region. Chosen over sample_237 to avoid a second lakeside scene. Prompt spelling "Faberg" is the dataset's; kept verbatim. |
| 5 | sample_116 | A young tree sprouts at the front of the house and grows quickly until it stands full-height in front of the facade. | tree grows; in front of house | hard (gradual) | no | Explicit location on bare ground before the facade; "sprouts and grows" tests the appearance window. |

Alternates (only via §9): easy `sample_236` (barn on hill crest); small `sample_237` (cairn on far bank); gradual `sample_102` (cactus next to mushroom).

Shortlisted and rejected: `sample_228` (trees already in front of the building), `sample_131` (sprouts already on the egg; animation), `sample_239` (target would be occluded by the tram), `sample_243` (region would be the whole frame), `sample_230` (crowded, non-static scene).

### 3.3 No-edit control prompts (optional arm; written 2026-11-18, before generation)

Same images, vanilla settings, same seed. Captions describe the scene as is and allow only natural motion; no new object, person, animal or weather event. Labelled with the same rubric, reported separately, never pooled with the edit arms. Baseline for preservation and for presence false positives; the queried object is labelled, not assumed absent.

| id | motion-neutral caption (no new object) |
|---|---|
| sample_240 | A quiet modern living room with a grey sofa and a glass partition; daylight shifts slowly through the window blinds. |
| sample_188 | A small round cafe table with two empty chairs by a sunny window; the light moves gently across the wooden floor. |
| sample_208 | A calm autumn lake bordered by bare trees and reeds; the water surface ripples slightly and the reflection shimmers. |
| sample_245 | An empty shopping cart standing in a supermarket aisle in front of refrigerated shelves; the camera holds steady with a subtle drift. |
| sample_116 | An old stone farmhouse behind a hedge, with a ploughed field in front and hills behind; clouds drift slowly across the sky. |

## 4. Intended regions

### 4.1 Rules
- Normalised `[x0,y0,x1,y1]` on the original image, origin top-left. Multiple regions (A, B) when the prompt permits; `free_placement` if no location can be inferred.
- The box covers the object's full extent at the end of the video (not at onset) plus ~5 % margin; no shadows or reflections.
- Drawn from the text in §4.3 only, on the reference image, before any video exists. No coordinates are suggested to annotators.
- Two annotators drew independently with `tools/draw_regions.py` (2026-11-18): A = author; B = MSc student, given only the tool, the five images and §4.3. IoU and disagreement type per region: `tools/compare_regions.py`. Resolution: IoU ≥ 0.5 → A's box is final; IoU < 0.5 → A redrew from the text (pass A2) and that box is final. Mean A–B IoU over 7 regions 0.49 (0.14–0.85).
- Generated frames: FramePack center-crops/resizes to a ~640-px bucket; the region is re-projected using the recorded crop.
- Placement of an object box: **in** if ≥ 50 % of its area lies inside the union of regions; **partial** 10–50 %; **out** otherwise.

### 4.2 Boxes

| id | region | annotator A | annotator B | IoU | disagreement type | final |
|---|---|---|---|---|---|---|
| sample_240 | A | `0.283,0.435,0.343,0.749` | `0.281,0.419,0.528,0.952` | 0.143 | horizontal extent: A took only the gap beside the partition; B took partition to sofa arm as written | `0.281,0.421,0.426,0.833` |
| sample_240 | B | `0.942,0.562,0.998,0.999` | `0.792,0.508,0.993,0.976` | 0.218 | where "the end of the chaise" lies | `0.812,0.534,1.000,0.999` |
| sample_188 | A | `0.295,0.027,0.806,0.970` | `0.163,0.002,0.948,0.989` | 0.623 | margin size; same chair | `0.295,0.027,0.806,0.970` |
| sample_188 | B | `0.002,0.090,0.302,0.924` | `0.002,0.002,0.445,0.982` | 0.577 | margin size; same chair | `0.002,0.090,0.302,0.924` |
| sample_208 | A | `0.002,0.182,0.997,0.467` | `0.002,0.240,0.997,0.495` | 0.724 | lower edge: waterline vs. slightly below | `0.002,0.182,0.997,0.467` |
| sample_245 | A | `0.459,0.630,0.813,0.896` | `0.302,0.379,0.802,0.834` | 0.278 | half-depth margin above the rim read differently; B also ran past the front panel | `0.400,0.479,0.801,0.865` |
| sample_116 | A | `0.398,0.323,0.844,0.787` | `0.386,0.302,0.863,0.765` | 0.854 | margin only | `0.398,0.323,0.844,0.787` |

Final boxes for 240 A, 240 B and 245 A are A2 redraws (IoU vs B: 0.45, 0.79, 0.59).

Agreement was high where the rule named one landmark (facade, shoreline) and low where it named two landmarks with clutter between them or a margin as a fraction of an object. Files: `data/regions/regions.json`, `data/regions/<id>_box.jpg`, overlays `data/regions/annotations/compare_A_B_<id>.jpg`.

### 4.3 Per-sample region descriptions (written 2026-11-18, before drawing)

**sample_240 — globe next to the sofa.** "Next to" means beside the sofa on the floor, not on it and not in front of it. Two valid regions.
- Region A (left end): the floor space between the left arm of the sofa and the glass partition to the bedroom. This is where the small side tables and the potted plant currently stand; the globe may replace or stand among them. Horizontal extent: from the glass partition to the left arm of the sofa. Vertical extent: from the floor up to the lower edge of the framed pictures on the wall.
- Region B (right end): the floor space between the right end of the chaise and the window wall. Horizontal extent: from the end of the chaise to the window blinds. Vertical extent: from the floor up to the top of the sofa backrest.

**sample_188 — person sits in the empty chair, picks up the book.** The object is a seated person. The region is the chair plus the volume a seated adult occupies above it. Two valid regions, because both chairs in the frame are empty; the fully visible right-hand chair is the primary one.
- Region A (right chair): horizontal extent: the chair's full width plus half a chair width on each side, to allow arms and shoulders. Vertical extent: from the floor at the chair legs up to roughly one chair height above the top of the backrest, to include the head.
- Region B (left chair, cut off by the frame edge): same construction, clipped at the left image border.
- The book on the table is not part of the placement region; whether the person picks it up is recorded under `object_correct` / notes, not placement.

**sample_208 — stag at the edge of the lake.** The only shoreline visible is the far bank; there is no near bank in the frame. One region.
- Region A (far shore strip): a horizontal strip across the full image width. Lower edge: slightly below the waterline, so that legs standing in shallow water are inside. Upper edge: roughly the mid-height of the tree trunks on the far bank, high enough to contain antlers of a stag standing on the bank. The reeds and the small evergreen are inside the strip.
- A stag standing in open water in the middle of the lake, or appearing in the reflection only, is **out**.

**sample_245 — egg in the shopping cart.** One region: the interior of the cart basket.
- Region A: horizontal extent: from the yellow front panel of the basket to the handle end of the basket (the handle bar itself excluded). Vertical extent: from the basket floor up to half a basket depth above the top rim, so that a "large" egg protruding above the rim is still inside. The wheels and the lower tray are excluded.

**sample_116 — tree sprouts in front of the house and grows to full height.** "At the front of the house" means between the facade and the camera, in the garden/hedge line or the edge of the ploughed field immediately before it. The region covers the full-grown tree. One region.
- Region A: horizontal extent: the width of the house facade, from the left corner of the main stone building to the right end of the lower right wing (the turret end). Vertical extent: lower edge a little below the hedge line, into the first rows of the ploughed field; upper edge at the roof ridge / chimney line of the main house, since "full-height in front of the facade" implies a tree about as tall as the house.
- A tree appearing beside the house (left of the main building, among the existing trees) or on the far hills is **out**. A tree whose crown rises above the roof ridge is still **in** if ≥ 50 % of its box lies inside the region.

## 5. Expected-appearance window

Defined on the 16 uniformly spaced extracted frames `f00..f15`, checked against the full video. **Onset** = first extracted frame in which the object is recognisable. **Persisting** = visible in ≥ 75 % of frames from onset to `f15` inclusive, and in `f15`. Late onset or non-persistence is a failure of that field, not "absent".

| id | window | reason |
|---|---|---|
| sample_240 | onset ≤ `f03` (25 %) and persisting | "appears" |
| sample_188 | onset ≤ `f03` and persisting; the sitting/reading action is labelled under `object_correct` / notes, not onset | person must be present early; the action is not part of the window |
| sample_208 | onset ≤ `f03` (25 %) and persisting | "appears" |
| sample_245 | onset ≤ `f03` (25 %) and persisting | "appears" |
| sample_116 | onset ≤ `f07` (50 %) and persisting; full height only by `f15` | "sprouts and grows"; a bare field at `f00` is expected |

The no-edit arm has no window; its presence label is recorded as observed.

## 6. Generation settings (identical across arms except γ)

FramePack-F1 `inference_f1_json.py`: `--seed 31337 --steps 25 --cfg 1.0 --gs 10.0 --rs 0.0 --teacache True`.
`--length`: 6.0 s if the hardware allows; otherwise the largest of 3.0 / 2.0 that completes the vanilla dry run (C3). `--gpu_mem`: set per machine. Both are identical across arms and recorded in every JSON sidecar.
Arms: vanilla `--control_scale 1.0`; AlignVid `--control_scale 1.35`; optional no-edit = vanilla settings with §3.3 prompts.
Hardware: RTX 5050 laptop (8 GB) or a free Colab/Kaggle GPU; the machine used is recorded in `logs/env_<machine>.txt`.

## 7. Human labels

Rubric: `protocol/label_rubric.md`, frozen with this file. Labelling is blind to arm (random codes; map in `labels/blind_map.csv`). Second labeller if available: raw agreement per field and Cohen's κ for presence, n stated.

## 8. Detector settings (fixed on dev examples, not pilot samples)

Fixed 2026-11-18 on three OmitI2V images that are not pilot samples (`measure/dev_thresholds.py`, outputs `data/detector_dev/`), each queried for a visibly present object and for the absent object its own prompt would add.

| Item | Value |
|---|---|
| Model | Grounding DINO tiny (§2), CPU |
| Box / text threshold | **0.45 / 0.35**. At 0.35 / 0.25 "stack of stones" gave two spurious boxes on sample_237; at 0.45 / 0.35 none, with present objects still detected at 0.72–0.94 |
| Frames per video | 16 (`f00..f15`) |
| Detector presence | detected in ≥ **k = 4** of 16 frames |
| Detector onset / persistence | first detected frame; detected frames / frames from onset to f15 |
| Detector placement | highest-scoring box on its best frame vs. the union of §4 regions, §4.1 rule |
| Pre-existing check | same query run on the reference image; a generated-frame box with IoU ≥ 0.5 to a reference box is flagged `pre_existing` and not counted |
| Dev images | sample_237 (boat present / stack of stones absent), sample_239 (tram / cable car), sample_102 (mushroom / cactus) |

Known failure mode: for an absent query the detector boxed the most salient existing object with high confidence ("cactus" → mushroom at 0.81; "cable car" → tram at 0.84), at both threshold pairs. Detector presence is therefore reported only alongside the human label and the pre-existing check.

VLM: none in the pilot. A Qwen2.5-VL-7B stand-in, if run later, gives per-question answers only, never pooled with the paper's protocol.

## 9. Deviations (dated)

| date | what changed | why |
|---|---|---|
| | | |
