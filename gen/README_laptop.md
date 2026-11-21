# Pilot generation package, round 1

Thank you for lending the GPU. This folder runs ten short image-to-video generations
(plus five optional ones) with FramePack-F1 and writes timing and memory logs. Nothing
is installed system-wide: a Python environment is created inside this folder, and the
model weights (about 35 GB) are downloaded into `external/AlignVid/models/FramePack/hf_download`.

## Before you start

1. Python 3.11 (3.10 or 3.12 also fine), installed for your user only: python.org,
   Windows installer, untick "for all users", tick "Add python.exe to PATH" and "py launcher".
2. About 40 GB free on the drive where this folder is.
3. Plugged in, and sleep disabled while plugged in (Settings > System > Power). The lid
   must stay open unless lid-close is set to "do nothing".
4. Nothing else heavy running on the GPU (games, video editors).

## Run

Double-click `gen\round1.bat` (or run it from a terminal). Leave the window open. It will:

- create the Python environment and install packages (10 to 20 minutes),
- download the model weights (35 GB, depends on your connection),
- write an environment report to `logs\env_<pc-name>.txt`,
- run one 2-second test video, then the real videos one by one. On an 8 GB laptop GPU
  expect roughly 20 to 40 minutes per 6-second video, so the full set can take most of a
  night. It can be stopped and restarted: finished videos are skipped.

If anything fails, the script stops by itself and still creates a zip of the logs.

## When it finishes

Upload the `results_<pc-name>_<date>.zip` file from this folder (it holds `outputs\` and
`logs\` only, no weights). If there is no zip, upload the `logs` folder.

## To free the disk afterwards

Delete this whole folder. The weights live inside it, nothing was installed elsewhere.
