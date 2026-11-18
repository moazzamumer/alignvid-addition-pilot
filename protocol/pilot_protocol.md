# Pilot protocol (pre-registration)

**Status:** DRAFT → mark FROZEN once complete
**Frozen on:** YYYY-MM-DD (Asia/Karachi)
**Frozen at commit:** `_______` (fill after the freeze commit; amend once)
**Author:** Moazzam Umer Gondal
**Second annotator:** MSc student (annotator B), drew from §4.3 text only

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

### 3.3 No-edit control prompts (optional arm; written 2026-11-18, before generation)

Same five reference images, vanilla settings (`--control_scale 1.0`), same seed as the other arms. Each caption describes the scene as it is and permits only natural motion; it introduces no object, person, animal or weather event. Labelled with the same rubric and reported in a separate table; never pooled with the edit arms. Purpose: a baseline for preservation (how much this model changes the scene when asked for nothing) and for presence false positives (the queried object must not be assumed absent; it is labelled).

| id | motion-neutral caption (no new object) |
|---|---|
| sample_240 | A quiet modern living room with a grey sofa and a glass partition; daylight shifts slowly through the window blinds. |
| sample_188 | A small round cafe table with two empty chairs by a sunny window; the light moves gently across the wooden floor. |
| sample_208 | A calm autumn lake bordered by bare trees and reeds; the water surface ripples slightly and the reflection shimmers. |
| sample_245 | An empty shopping cart standing in a supermarket aisle in front of refrigerated shelves; the camera holds steady with a subtle drift. |
| sample_116 | An old stone farmhouse behind a hedge, with a ploughed field in front and hills behind; clouds drift slowly across the sky. |

## 4. Intended regions

### 4.1 Rules
- Box(es) in normalised `[x0,y0,x1,y1]` on the original image (x right, y down, 0–1); multiple boxes allowed when the prompt permits, labelled A, B; `free_placement` tag if no location can be inferred (no box).
- The box covers the **full extent the object is expected to occupy at the end of the video**, not at onset (relevant for "grows" prompts), plus a margin of roughly 5 % of the image side. It does not include shadows or reflections.
- Boxes are drawn **from the text in §4.3 only**, on the reference image, before any video is generated or viewed. No pixel coordinates are suggested to the annotator.
- Annotation: two annotators drew independently from the text in §4.3 using `tools/draw_regions.py`: A = the author (2026-11-18); B = an independent second annotator (MSc student; recorded in the header), who received only the tool, the five reference images and §4.3 as a self-contained kit, with no verbal guidance on box placement. IoU per region and the type of disagreement were recorded (`tools/compare_regions.py`). **Resolution rule:** for regions with IoU ≥ 0.5 (188 A/B, 208 A, 116 A) annotator A's box is final. For regions with IoU < 0.5 (240 A, 240 B, 245 A) annotator A re-read the §4.3 text and redrew the region (pass "A2"); that box is final. Mean A–B IoU over 7 regions: 0.49 (min 0.14, max 0.85).
- Mapping to generated frames: FramePack center-crops/resizes to a ~640-px bucket; the region is re-projected onto the first generated frame using the recorded crop.
- Placement rule for a detected/judged object box: **in** if ≥ 50 % of its area lies inside the (union of) region(s); **partial** if 10–50 %; **out** otherwise.

### 4.2 Boxes

Normalised `[x0,y0,x1,y1]`. A = author pass 1; B = second annotator; final = A's box, or A's redraw (A2) where stated.

| id | region | annotator A | annotator B | IoU | disagreement type | final |
|---|---|---|---|---|---|---|
| sample_240 | A | `0.283,0.435,0.343,0.749` | `0.281,0.419,0.528,0.952` | 0.143 | horizontal extent: A took only the gap beside the partition; B took partition-to-sofa-arm including the side tables, as written. Final: A redrew from the text (A2). | `0.281,0.421,0.426,0.833` |
| sample_240 | B | `0.942,0.562,0.998,0.999` | `0.792,0.508,0.993,0.976` | 0.218 | where 'the end of the chaise' lies: A drew only the strip at the blinds; B started at the chaise end, as written. Final: A redrew from the text (A2). | `0.812,0.534,1.000,0.999` |
| sample_188 | A | `0.295,0.027,0.806,0.970` | `0.163,0.002,0.948,0.989` | 0.623 | margin size only: same chair; B used a larger margin and ran to the top of the frame. Final: A's box. | `0.295,0.027,0.806,0.970` |
| sample_188 | B | `0.002,0.090,0.302,0.924` | `0.002,0.002,0.445,0.982` | 0.577 | margin size only: same chair, clipped at the frame edge. Final: A's box. | `0.002,0.090,0.302,0.924` |
| sample_208 | A | `0.002,0.182,0.997,0.467` | `0.002,0.240,0.997,0.495` | 0.724 | lower edge only: B put it at the waterline, A slightly below it as written. Final: A's box. | `0.002,0.182,0.997,0.467` |
| sample_245 | A | `0.459,0.630,0.813,0.896` | `0.302,0.379,0.802,0.834` | 0.278 | fraction-based margin read differently: A omitted the half-depth margin above the rim; B included it but ran past the yellow panel and stopped above the basket floor. Final: A redrew from the text (A2). | `0.400,0.479,0.801,0.865` |
| sample_116 | A | `0.398,0.323,0.844,0.787` | `0.386,0.302,0.863,0.765` | 0.854 | near-identical; small margin difference. Final: A's box. | `0.398,0.323,0.844,0.787` |

A2 redraws (2026-11-18): 240 A `0.281,0.421,0.426,0.833` (IoU vs A 0.32, vs B 0.45); 240 B `0.812,0.534,1.000,0.999` (vs A 0.28, vs B 0.79); 245 A `0.400,0.479,0.801,0.865` (vs A 0.48, vs B 0.59).

Observation for the revised protocol: agreement was high where the rule named a single landmark (facade, shoreline) and low where it named two landmarks with clutter between them (partition–sofa arm, chaise–window) or a margin expressed as a fraction of an object (half a basket depth). Overlays: `data/regions/annotations/compare_A_B_<id>.jpg`.

Machine-readable: `data/regions/regions.json`. Previews: `data/regions/<id>_box.jpg`.

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

Defined on the 16 uniformly spaced extracted frames `f00..f15` (and checked against the full video). **Onset** = first extracted frame in which the object is recognisable. **Persisting** = visible in ≥ 75 % of extracted frames from onset to `f15` inclusive, and visible in `f15`. A video whose onset is later than the window, or which does not persist, is scored as a failure of the corresponding field, not as "absent", if the object does appear at some point.

| id | window | reason |
|---|---|---|
| sample_240 | onset ≤ `f03` (≤ 25 %) and persisting | "appears": instantaneous prompt |
| sample_188 | onset ≤ `f03` (≤ 25 %) and persisting; the sitting/reading action may complete later and is labelled under `object_correct` / notes, not under onset | "sits down": the person must be present early; the action sequence is not part of the presence window |
| sample_208 | onset ≤ `f03` (≤ 25 %) and persisting | "appears": instantaneous prompt |
| sample_245 | onset ≤ `f03` (≤ 25 %) and persisting | "appears": instantaneous prompt |
| sample_116 | onset ≤ `f07` (≤ 50 %) and persisting; full height is not required until `f15` | "sprouts and grows": gradual prompt; a bare field at `f00` is expected and not penalised |

The no-edit arm has no window; its presence label is recorded as observed.

## 6. Generation settings (identical across arms except γ)

FramePack-F1 `inference_f1_json.py`: `--seed 31337 --length ___ --steps 25 --cfg 1.0 --gs 10.0 --rs 0.0 --gpu_mem ___ --teacache True`.
Arms: vanilla `--control_scale 1.0`; AlignVid `--control_scale 1.35`; (optional) no-edit = vanilla settings with §3.3 prompts.
Hardware intended: ______ (fill actual in the report).

## 7. Human labels

Rubric: `protocol/label_rubric.md` (frozen with this file). Labelling is blind to arm (random codes, map sealed in `labels/blind_map.csv`). Second labeller on all videos if available; report raw agreement per field and Cohen's κ for presence with n stated.

## 8. Detector settings (fixed on dev examples, not pilot samples)

Fixed 2026-11-18 on three OmitI2V reference images that are not pilot samples, before any video was generated (`measure/dev_thresholds.py`; outputs in `data/detector_dev/`). Each image was queried for an object visibly present and for the object its own prompt would add, which is absent.

| Item | Value |
|---|---|
| Model | Grounding DINO tiny, HF `IDEA-Research/grounding-dino-tiny`, `model.safetensors` sha256 prefix `1a2412ef99bd74bc`; transformers 5.x, CPU |
| Box / text threshold | **0.45 / 0.35** (0.35 / 0.25 produced two spurious boxes for "stack of stones" on sample_237; 0.45 / 0.35 produced none, with the present objects still detected at 0.72–0.94) |
| Frames per video | 16 (`f00..f15`) |
| Detector presence | detected in ≥ **k = 4** of 16 frames |
| Detector onset | first frame with a detection; detector persistence = detected frames / frames from onset to f15 |
| Detector placement | highest-scoring box per frame vs. the union of §4 regions, §4.1 rule, on the frame with the highest score |
| Pre-existing check | the detector is also run on the reference image with the same query; a generated-frame box with IoU ≥ 0.5 to a reference-image box is flagged `pre_existing` and not counted toward presence |
| Dev images | sample_237 (boat present / stack of stones absent), sample_239 (tram / cable car), sample_102 (mushroom / cactus) |

Known failure mode recorded before generation: for an absent query the detector returned the most salient existing object with high confidence (sample_102, query "cactus", boxed the mushroom at 0.81 at both threshold pairs; sample_239, query "cable car", boxed the tram at 0.84). Thresholds do not remove this. Detector presence is therefore reported only alongside the human label and the pre-existing check, never on its own.

VLM: none in the pilot. If a Qwen2.5-VL-7B stand-in is run later (step D4), per-question answers only; never pooled with the paper's protocol.

## 9. Deviations (dated)

| date | what changed | why |
|---|---|---|
| | | |
