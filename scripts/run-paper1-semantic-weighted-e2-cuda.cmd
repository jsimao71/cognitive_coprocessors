@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
set "PYTHON=%USERPROFILE%\.venvs\ccpu-cuda\Scripts\python.exe"
set "PYTHONPATH=%REPO_ROOT%\src"
set "CONFIG=%REPO_ROOT%\configs\paper1\asl_matrix_q0_semantic_weighted_e2_qwen_lora_cuda.json"
set "DATA=%REPO_ROOT%\artifacts\paper1\asl_matrix_v1\qwen_data\q0_seed11"
set "EVAL_SOURCE=%REPO_ROOT%\artifacts\paper1\asl_pilot_v1\data\eval\original.jsonl"
set "TRAIN_SPLIT=%REPO_ROOT%\artifacts\paper1\asl_pilot_500_v1\data\sft\train_450.jsonl"
set "OUTPUT=%REPO_ROOT%\artifacts\paper1\e3_v2\m0_5_semantic_weighted_e2_cuda"
set "RUN=%OUTPUT%\qwen_run"
set "EVALUATION=%OUTPUT%\eval"

if not exist "%PYTHON%" (
    echo CUDA Python is unavailable: %PYTHON% 1>&2
    exit /b 1
)

if not "%~1"=="" call :wait_for_pid %~1
if errorlevel 1 exit /b 1

pushd "%REPO_ROOT%"
if not exist "%RUN%\training_report.json" (
    "%PYTHON%" -u -m ccpu paper1 train-lora ^
        --config "%CONFIG%" ^
        --model "Qwen/Qwen3-0.6B" ^
        --train "%DATA%\train.jsonl" ^
        --dev "%DATA%\dev.jsonl" ^
        --output-dir "%RUN%"
    if errorlevel 1 goto :failed
)

if not exist "%EVALUATION%\summary.json" (
    "%PYTHON%" -u -m ccpu paper1 run-asl-pilot ^
        --eval "%EVAL_SOURCE%" ^
        --train-split "%TRAIN_SPLIT%" ^
        --config "%REPO_ROOT%\configs\paper1\asl_pilot_qwen_base_cuda.json" ^
        --adapter-path "%RUN%\adapter" ^
        --adapter-id "Qwen3-0.6B-ASL-Matrix-Q0-SemanticWeighted-E2-r8-seed11-cuda" ^
        --condition lora ^
        --shots 0 ^
        --output-dir "%EVALUATION%" ^
        --checkpoint-every 1
    if errorlevel 1 goto :failed
)
popd
exit /b 0

:wait_for_pid
set "WAIT_PID=%~1"
:wait_loop
tasklist /FI "PID eq %WAIT_PID%" /NH 2>nul | findstr /R /C:"[ ]%WAIT_PID%[ ]" >nul
if errorlevel 1 exit /b 0
echo Waiting for process %WAIT_PID% to release CUDA...
ping -n 31 127.0.0.1 >nul
goto :wait_loop

:failed
popd
exit /b 1
