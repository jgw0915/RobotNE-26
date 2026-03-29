@echo off
REM #NewFeature: Convenience launcher for the headless F1 challenge benchmark script.
setlocal

cd /d "%~dp0"
python benchmark_f1_challenge.py %*

endlocal
