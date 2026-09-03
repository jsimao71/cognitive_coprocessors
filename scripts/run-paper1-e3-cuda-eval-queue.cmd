@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
set "PYTHON=%USERPROFILE%\.venvs\ccpu-cuda\Scripts\python.exe"
set "PYTHONPATH=%REPO_ROOT%\src"
set "ASL_EVAL=%REPO_ROOT%\artifacts\paper1\asl_pilot_v1\data\eval\original.jsonl"
set "ASL_TRAIN=%REPO_ROOT%\artifacts\paper1\asl_pilot_500_v1\data\sft\train_450.jsonl"
set "M06=%REPO_ROOT%\artifacts\paper1\e3_v2\m0_6_semantic_ranked_cuda"
set "F4=%REPO_ROOT%\artifacts\paper1\e3_v2\m1_bottleneck"

if not exist "%PYTHON%" (
    echo CUDA Python is unavailable: %PYTHON% 1>&2
    exit /b 1
)

if not "%~1"=="" call :wait_for_pid %~1
if errorlevel 1 exit /b 1

pushd "%REPO_ROOT%"
if not exist "%M06%\qwen_run\training_report.json" (
    echo M0.6 training report is missing. 1>&2
    goto :failed
)

if not exist "%M06%\eval\summary.json" (
    "%PYTHON%" -u -m ccpu paper1 run-asl-pilot ^
        --eval "%ASL_EVAL%" ^
        --train-split "%ASL_TRAIN%" ^
        --config "%REPO_ROOT%\configs\paper1\asl_pilot_qwen_base_cuda.json" ^
        --adapter-path "%M06%\qwen_run\adapter" ^
        --adapter-id "Qwen3-0.6B-ASL-Matrix-Q0-SemanticRanked-r8-seed11-cuda" ^
        --condition lora ^
        --shots 0 ^
        --output-dir "%M06%\eval" ^
        --checkpoint-every 1
    if errorlevel 1 goto :failed
)

if not exist "%F4%\eval_cuda\summary.json" (
    "%PYTHON%" -u -m ccpu.paper1.e3 run-bottleneck ^
        --eval "%REPO_ROOT%\artifacts\paper1\e3_v2\bottleneck_v1\sft\test.jsonl" ^
        --config "%REPO_ROOT%\configs\paper1\e3_f4_qwen_eval_cuda.json" ^
        --adapter-path "%F4%\qwen_run\adapter" ^
        --adapter-id "Qwen3-0.6B-E3-F4-L0-r8-seed11" ^
        --objective-id L0 ^
        --output-dir "%F4%\eval_cuda" ^
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
