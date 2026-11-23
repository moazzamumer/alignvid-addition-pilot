# Progress log

| date | step | note |
|---|---|---|
| 2026-11-18 | A1–A2 | Private repository; AlignVid added as submodule `external/AlignVid` at commit `9f3e48d`; protocol drafted from the template. |
| 2026-11-18 | A3–A4 | OmitI2V snapshot (367 records; addition/object 73) indexed with `tools/inspect_meta.py`; counts and checksum in protocol §2–§3. |
| 2026-11-18 | A5 | Five samples chosen by the §3.1 rules: 240, 188, 208, 245, 116 (245 replaced 237 to avoid two lakeside scenes); alternates and rejects in §3.2. |
| 2026-11-18 | A6 | Intended regions: rules and per-sample descriptions written first; boxes drawn independently by two annotators; IoU and disagreement type recorded; final boxes in `data/regions/regions.json`. |
| 2026-11-18 | A7–A8 | Expected-appearance windows (§5), no-edit control captions (§3.3), label rubric and `labels/labels_template.csv`. |
| 2026-11-18 | A9 | Grounding DINO thresholds fixed on three non-pilot images (0.45 / 0.35, k = 4 of 16) with a pre-existing check against the reference image; failure mode noted in §8. |
| 2026-11-18 | A10 | Protocol frozen at commit `ec4d9e4`. Later changes are dated rows in §9. |
| 2026-11-19 | B1–B4 | Measurement code: frame extraction and contact sheets, detector with region re-projection through FramePack's resize and crop, panel builder, run logger (wall time, peak GPU and RAM, versions). Tested on synthetic clips. |
| 2026-11-22 | C1–C2 | Generation on the Kaggle free tier (Tesla T4 15 GB, 31 GB RAM, torch 2.11 cu128, Python 3.11). `gen/run_pilot.py` runs one FramePack process per video with a log per attempt; three source patches were needed before the first dry run (`gen/patches/README.md`). |
| 2026-11-22 | C3 | Dry runs on sample_240 at 2 s. bf16: out of memory in the attention fallback, fixed by chunking; then 8358 s per clip, too slow for the quota. fp16: black clip from overflow; with an fp32 residual stream (three sites found by `gen/nan_probe.py`) 1546 s and PSNR 37.1 dB against the bf16 clip. fp16 adopted for all arms; §9 updated. |
| 2026-11-22/23 | C4 | Session A: length probe chose 6.0 s, five vanilla videos. Session B: five AlignVid videos with A's state restored. 55–62 min per video, GPU peak 13.0–13.2 GB, RAM 28–29 GB, no failures, 9 h 49 min of quota in total. Outputs and logs under `outputs/<arm>/` and `logs/kaggle/`; frames extracted and detector run on both arms. |
| 2026-11-23 | D | Blind kit (`tools/make_blind_kit.py`, seeded codes V01–V10); labeller A labelled all ten videos. Second labeller not available; no-edit arm not generated. |
| 2026-11-23 | E | `report/build_report.py` produces the pilot report and the revised protocol as PDFs and CSV tables from the repository state; README added. |
