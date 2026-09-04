@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
if not defined CCPU_PYTHON set "CCPU_PYTHON=python"
set "PYTHONPATH=%REPO_ROOT%\src"
set "DATA=%REPO_ROOT%\artifacts\paper1\gsm8k_scale_v1\g1_f0_4500"
set "REPLICATION=%DATA%\replications\seed23_directml"
set "RUN=%REPLICATION%\qwen_run"
set "EVALUATION=%REPLICATION%\historical_test_eval"

"%CCPU_PYTHON%" -c "import torch_directml; print(torch_directml.device())" >nul
if errorlevel 1 (
    echo DirectML Python environment is unavailable: %CCPU_PYTHON% 1>&2
    exit /b 1
)
if not exist "%DATA%\eval\manifest.json" (
    echo GSM8K-only eval freeze is unavailable. 1>&2
    exit /b 1
)

pushd "%REPO_ROOT%"
if not exist "%RUN%\training_report.json" (
    "%CCPU_PYTHON%" -u -m ccpu paper1 train-lora ^
        --config "%REPO_ROOT%\configs\paper1\e3_g1_gsm8k_f0_l0_qwen_lora_directml_seed23.json" ^
        --model "Qwen/Qwen3-0.6B" ^
        --train "%DATA%\train.jsonl" ^
        --dev "%DATA%\eval\dev.jsonl" ^
        --output-dir "%RUN%"
    if errorlevel 1 goto :failed
)

if not exist "%EVALUATION%\summary.json" (
    "%CCPU_PYTHON%" -u -m ccpu paper1 run-asl-pilot ^
        --eval "%DATA%\eval\test.jsonl" ^
        --train-split "%DATA%\train.jsonl" ^
        --config "%REPO_ROOT%\configs\paper1\asl_pilot_qwen_base_directml.json" ^
        --adapter-path "%RUN%\adapter" ^
        --adapter-id "Qwen3-0.6B-G1-GSM8K-F0-L0-r8-seed23-directml" ^
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
