# Progress log

| date | step | note |
|---|---|---|
| 2026-11-18 | A1 | Repo created and pushed (private): https://github.com/moazzamumer/alignvid-addition-pilot. First commit `f2cfc08` "pre-registration scaffold". |
| 2026-11-18 | A2 | AlignVid added as submodule `external/AlignVid` at commit `9f3e48d` (2026-05-29, single commit). Hash recorded in `protocol/pilot_protocol.md` §2 Versions. Working protocol created from template (status DRAFT). |
| 2026-11-18 | A3 | OmitI2V downloaded to `data/OmitI2V` (gitignored): 370 files, 273 MB, 367 records (addition 129 / deletion 125 / modification 113; addition/object = 73). `meta.json` sha256 recorded in protocol §2. Schema note: edit type is `main-category`, `type` is image source. Local Python is 3.11.2, not 3.10. |
| 2026-11-18 | A4 | `tools/inspect_meta.py` written and run; `data/addition_object_index.csv` has 73 rows. Counts written to protocol §3. Quirks: sample_121 prompt/key mismatch; sample_199 empty question category; sample_6 is dust, not an object. |
