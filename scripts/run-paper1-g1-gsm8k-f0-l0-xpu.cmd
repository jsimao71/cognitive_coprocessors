@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
set "PYTHON=%USERPROFILE%\.venvs\modal-llm-xpu\Scripts\python.exe"
set "PYTHONPATH=%REPO_ROOT%\src"
set "DATA=%REPO_ROOT%\artifacts\paper1\gsm8k_scale_v1\g1_f0_4500"
set "RUN=%DATA%\qwen_run"
set "EVALUATION=%DATA%\historical_test_eval"

if not exist "%PYTHON%" (
    echo XPU Python is unavailable: %PYTHON% 1>&2
    exit /b 1
)
if not exist "%DATA%\eval\manifest.json" (
    echo GSM8K-only eval freeze is unavailable. Run prepare-paper1-gsm8k-g1.cmd. 1>&2
    exit /b 1
)

pushd "%REPO_ROOT%"
if not exist "%RUN%\training_report.json" (
    "%PYTHON%" -u -m ccpu paper1 train-lora ^
        --config "%REPO_ROOT%\configs\paper1\e3_g1_gsm8k_f0_l0_qwen_lora_xpu.json" ^
        --model "Qwen/Qwen3-0.6B" ^
        --train "%DATA%\train.jsonl" ^
        --dev "%DATA%\eval\dev.jsonl" ^
        --output-dir "%RUN%"
    if errorlevel 1 goto :failed
)

if not exist "%EVALUATION%\summary.json" (
    "%PYTHON%" -u -m ccpu paper1 run-asl-pilot ^
        --eval "%DATA%\eval\test.jsonl" ^
        --train-split "%DATA%\train.jsonl" ^
        --config "%REPO_ROOT%\configs\paper1\asl_pilot_qwen_base_xpu.json" ^
        --adapter-path "%RUN%\adapter" ^
        --adapter-id "Qwen3-0.6B-G1-GSM8K-F0-L0-r8-seed11" ^
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
