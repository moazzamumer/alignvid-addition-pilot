# AlignVid object-addition pilot

Pre-registered five-sample feasibility pilot: does a measurement pipeline for **object addition** in image-to-video (presence, placement, persistence, preservation) run end to end on accessible hardware, and where do labels, tools and compute break down? Backbone FramePack-F1, arms vanilla (`--control_scale 1.0`) and AlignVid (`--control_scale 1.35`), one seed, OmitI2V `addition/object` samples. The pilot validates the pipeline; it does not estimate a performance difference.

## Deliverables

- `report/pilot_report.pdf`: header and versions, panel (reference with intended regions, instruction, 6-frame strips per arm, human labels, detector summary), tables T1 human labels, T2 human vs detector, T4 runtime and memory, T5 failures, "where it broke down", deviations from the frozen protocol.
- `report/revised_protocol.pdf`: revised short protocol and measured cost table.
- `report/tables/*.csv`: the same tables as data.

## Layout

| Path | Content |
|---|---|
| `protocol/pilot_protocol.md`, `protocol/label_rubric.md` | pre-registration (frozen 2026-11-18, commit `ec4d9e4`); §9 holds dated deviations |
| `data/` | sample index and selection, intended regions (`regions/`), prompt files, detector queries, detector dev-threshold outputs |
| `gen/` | generation orchestrator (`run_pilot.py`), recorded source patches for FramePack on the Tesla T4 (`patches/`), fp16 overflow probe |
| `kaggle/pilot_kaggle.ipynb` | headless Kaggle notebook that runs `gen/run_pilot.py` |
| `measure/` | frame extraction, Grounding DINO check, panel builder, system/run logger |
| `outputs/<arm>/` | per-video sidecars (wall time, peak memory, versions) and detector results; videos and frames are kept locally |
| `logs/kaggle/` | every generation attempt (incl. failed and slow dry runs), environment reports, applied patches, executed notebooks |
| `labels/` | blind map, label skeletons and labels; `tools/make_blind_kit.py` builds the labelling kit |
| `report/` | report builder (`build_report.py`), prose sections, panels, PDFs and CSV tables |
| `progress.md` | dated log of what was done |
| `external/AlignVid` | the AlignVid repository as a submodule (commit `9f3e48d`), unmodified; patches are applied at run time and logged |

## Reproduce

```
python measure/extract_frames.py outputs/<arm>/*.mp4 --out-dir outputs/<arm>/frames
python measure/detect.py --arm <arm> --ids sample_240 sample_188 sample_208 sample_245 sample_116
python tools/make_blind_kit.py --arms vanilla alignvid --labellers A B
python measure/panel.py --arms vanilla alignvid --labels labels/labels_A.csv --out report/panels
python report/build_report.py
```

Generation: see the docstring of `gen/run_pilot.py` and `kaggle/pilot_kaggle.ipynb`.
