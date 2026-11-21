@echo off
REM AlignVid pilot, round 1: environment, FramePack install, dry run, generation.
REM Run from anywhere; everything happens inside the folder this file lives in.
REM Needs: Python 3.10, 3.11 or 3.12 installed for the current user, internet, ~40 GB free disk.
setlocal
cd /d "%~dp0\.."
if not exist logs mkdir logs
echo [round1] working folder: %cd%
echo [round1] console output is also written to logs\round1_console.txt

if not exist gen\.venv-gen\Scripts\python.exe (
  echo [round1] creating Python environment gen\.venv-gen ...
  py -3.11 -m venv gen\.venv-gen 2>nul || py -3.12 -m venv gen\.venv-gen 2>nul || py -3.10 -m venv gen\.venv-gen 2>nul || python -m venv gen\.venv-gen
)
if not exist gen\.venv-gen\Scripts\python.exe (
  echo [round1] ERROR: could not create a Python environment. Install Python 3.11 from python.org (Install for me only) and rerun.
  pause & exit /b 1
)
set PY=gen\.venv-gen\Scripts\python.exe
%PY% -c "import sys; v=sys.version_info[:2]; print('[round1] python', sys.version.split()[0]); sys.exit(0 if (3,10)<=v<=(3,12) else 1)"
if errorlevel 1 (
  echo [round1] ERROR: Python must be 3.10-3.12. Delete gen\.venv-gen, install Python 3.11 and rerun.
  pause & exit /b 1
)

echo [round1] installing torch (CUDA 12.8 build, ~3 GB download) ... see logs\pip_torch.txt
%PY% -m pip install --upgrade pip >> logs\pip_torch.txt 2>&1
%PY% -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul || %PY% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 >> logs\pip_torch.txt 2>&1
%PY% -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul || %PY% -m pip install torch torchvision >> logs\pip_torch.txt 2>&1
echo [round1] installing FramePack requirements ... see logs\pip_requirements.txt
%PY% -m pip install -r gen\requirements-laptop.txt >> logs\pip_requirements.txt 2>&1
if errorlevel 1 echo [round1] WARNING: some requirements failed to install; continuing so the env log is still written.

echo [round1] starting the pilot (env check, model download ~35 GB, dry run, videos). Leave this window open.
%PY% gen\run_pilot.py %* >> logs\round1_console.txt 2>&1
type logs\run_pilot.log | findstr /R "stage done FAILED chosen packed stopped finished"
echo.
echo [round1] finished. Please upload the results_*.zip file from this folder (or the logs folder if there is no zip).
pause
