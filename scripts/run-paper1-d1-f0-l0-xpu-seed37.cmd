@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
set "PYTHON=%USERPROFILE%\.venvs\modal-llm-xpu\Scripts\python.exe"
set "PYTHONPATH=%REPO_ROOT%\src"
set "DATA=%REPO_ROOT%\artifacts\paper1\e3_v2\d1_f0_v1"
set "D0=%REPO_ROOT%\artifacts\paper1\asl_pilot_500_v1\data\sft"
set "EVAL_SOURCE=%REPO_ROOT%\artifacts\paper1\asl_pilot_v1\data\eval\original.jsonl"
set "RUN=%DATA%\replications\seed37\qwen_run"
set "EVALUATION=%DATA%\replications\seed37\eval"

if not exist "%PYTHON%" (
    echo XPU Python is unavailable: %PYTHON% 1>&2
    exit /b 1
)

pushd "%REPO_ROOT%"
if not exist "%RUN%\training_report.json" (
    "%PYTHON%" -u -m ccpu paper1 train-lora ^
        --config "%REPO_ROOT%\configs\paper1\e3_d1_f0_l0_qwen_lora_xpu_seed37.json" ^
        --model "Qwen/Qwen3-0.6B" ^
        --train "%DATA%\train.jsonl" ^
        --dev "%D0%\dev.jsonl" ^
        --output-dir "%RUN%"
    if errorlevel 1 goto :failed
)

if not exist "%EVALUATION%\summary.json" (
    "%PYTHON%" -u -m ccpu paper1 run-asl-pilot ^
        --eval "%EVAL_SOURCE%" ^
        --train-split "%DATA%\train.jsonl" ^
        --config "%REPO_ROOT%\configs\paper1\asl_pilot_qwen_base_xpu.json" ^
        --adapter-path "%RUN%\adapter" ^
        --adapter-id "Qwen3-0.6B-E3-D1-F0-L0-r8-seed37" ^
        --condition lora ^
        --shots 0 ^
        --output-dir "%EVALUATION%" ^
        --checkpoint-every 1
    if errorlevel 1 goto :failed
)
popd
exit /b 0

:failed
popd
exit /b 1
