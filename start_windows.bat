@echo off
cd /d "%~dp0"
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
   py -3 -m pip install -r requirements.txt
   py -3 usb_beam_profiler.py
) else (
   python -m pip install -r requirements.txt
   python usb_beam_profiler.py
)
pause
