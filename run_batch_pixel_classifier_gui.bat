@echo off
cd /d "%~dp0"
set "PYTHON_EXE=python"
if exist "C:\ProgramData\miniforge3\envs\smartMicroscopy\python.exe" set "PYTHON_EXE=C:\ProgramData\miniforge3\envs\smartMicroscopy\python.exe"
"%PYTHON_EXE%" src\run_batch_pixel_classifier_gui.py
pause
