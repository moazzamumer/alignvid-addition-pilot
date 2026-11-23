# Revised short protocol and cost table

Pilot-based revision, 2026-11-23. Replaces the pre-registered pilot protocol (frozen 2026-11-18, commit `ec4d9e4`) for any larger run; the pilot's own deviations are listed in the pilot report.

## 1. Scope

Three primary outcomes per video, human-labelled: **presence**, **placement** and **persistence** of the requested addition, each defined against a per-sample expected-appearance window. **Preservation** is a human rubric score (3 / 2 / 1 with a note) read together with presence; any pixel or feature metric for preservation comes later, with one common exclusion rule across arms (the intended region, not a treatment-specific object mask) and a sensitivity check over mask size. No motion mechanism is tested; any stratification by scene motion is descriptive.

## 2. Data

- OmitI2V `addition/object` subset (73 samples in the 2026-11-18 snapshot; 129 for all addition types). Selection by written rules before any output is seen: a placeable object, a reference image without an existing instance, a mix of easy and hard additions (small, gradual, contact with an existing subject), no duplicate scene types.
- Intended regions drawn before generation by two annotators from a per-sample written description; multiple valid regions allowed when the prompt permits; free-placement prompts kept and reported separately; IoU and disagreement type recorded; resolution rule fixed in advance. Draw regions to cover the full extent the object may occupy (the pilot's egg exceeded its cart region).
- Expected-appearance window per sample, stated in frames of the 16-frame sheet (f00–f15), written before generation. Record the actual clip length and frame rate the backbone produces.

## 3. Arms

Vanilla (γ = 1.0) and AlignVid (γ = 1.35), identical settings and seeds. Three seeds per sample for the 40-sample study. A matched no-edit video per sample and seed (scene prompt, no object) as a separately labelled diagnostic for spontaneous additions and baseline scene change; never pooled with the edit arms. DyMoS: future work only; its repository now ships Wan2.1 code (github.com/shoxedqBbB/DyMoS, checked 2026-11), but integration with AlignVid would need matched checkpoints, schedulers, preprocessing and attention implementations.

## 4. Ground truth

Humans are the reference for every tool. Blind labelling (random codes, arms hidden), two labellers, raw agreement per field and κ for presence. Rubric changes from the pilot: a rule for partially visible objects (count as present, record the part and set object_correct accordingly); `transient` gets an n.a. value; presence is always reported with preservation and within_window; one derived field "within window and persisting" for readability. Before setting expected answers for any judge audit, humans label source presence (reference image) and generated presence and location; false positives are reported among human-confirmed negatives and false negatives among confirmed positives, with denominators. The published OmitI2V protocol is reported separately from any boxed or paired-question protocol.

## 5. Tools

Grounding DINO with thresholds and k fixed on development images that are not study samples (pilot: 0.45 / 0.35, k = 4 of 16); the pre-existing check against the reference image stays mandatory. Report detector-vs-human disagreements per field, tracking failures and unscorable cases. Placement rule: define on the object's support or on the share of the region covered, not only on the share of the detector box inside the region. A VLM, if used, is a stand-in for the paper's judge and gives per-question answers compared with human labels; its scores are not pooled with the paper's.

## 6. Analysis language

Associative only. Report counts per field and arm with denominators; no causal statements about motion, confounds or mechanisms; no "first" or taxonomy claims.

## 7. Cost table

Measured on the pilot hardware (Kaggle free tier, Tesla T4 15 GB, 31 GB RAM, FramePack-F1 in fp16 with fp32 residual, `--gpu_mem 6`, `--length 6.0`):

{{COST_TABLE}}

Which rows fit free tiers: the pilot itself (10 h) fits one week of Kaggle quota. Rows (a) and (b) need 7–8 weeks of the full weekly quota in 12 h sessions, which is feasible but slow and occupies the quota entirely; adding the no-edit arm adds about half again. Colab free is excluded by RAM (12.7 GB against a 28–29 GB peak). On a lab GPU that runs bf16 natively (Ampere or newer), FramePack's published figure of about 1.5 s per frame with TeaCache on an RTX 4090 would put a 181-frame video near 5 min, so rows (a) and (b) would each be about 20 GPU-hours; this is the authors' figure, not measured here.
