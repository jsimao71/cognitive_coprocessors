@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
set "PYTHON=%USERPROFILE%\.venvs\ccpu-cuda\Scripts\python.exe"
set "PYTHONPATH=%REPO_ROOT%\src"
set "DATA=%REPO_ROOT%\artifacts\paper1\e3_v2\f4_l2_preference_v1"
set "TEST=%REPO_ROOT%\artifacts\paper1\e3_v2\bottleneck_v1\sft\test.jsonl"
set "OUTPUT=%REPO_ROOT%\artifacts\paper1\e3_v2\m3_f4_semantic_ranked_cuda"
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
        --config "%REPO_ROOT%\configs\paper1\e3_f4_l2_qwen_lora_cuda.json" ^
        --model "Qwen/Qwen3-0.6B" ^
        --train "%DATA%\train.jsonl" ^
        --dev "%DATA%\dev.jsonl" ^
        --output-dir "%RUN%"
    if errorlevel 1 goto :failed
)

if not exist "%EVALUATION%\summary.json" (
    "%PYTHON%" -u -m ccpu.paper1.e3 run-bottleneck ^
        --eval "%TEST%" ^
        --config "%REPO_ROOT%\configs\paper1\e3_f4_qwen_eval_cuda.json" ^
        --adapter-path "%RUN%\adapter" ^
        --adapter-id "Qwen3-0.6B-E3-F4-L2-r8-seed11-cuda" ^
        --objective-id L2 ^
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
