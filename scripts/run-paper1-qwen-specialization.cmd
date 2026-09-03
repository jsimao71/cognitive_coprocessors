@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
set "PYTHON=C:\Users\j.simao\.venvs\modal-llm-xpu\Scripts\python.exe"
set "PYTHONPATH=%REPO_ROOT%\src"
set "DATA=%REPO_ROOT%\artifacts\paper1\asl_matrix_v1\qwen_data\q1_seed11"
set "SOURCE=%REPO_ROOT%\artifacts\paper1\asl_matrix_v1\data\source"
set "RUNS=%REPO_ROOT%\artifacts\paper1\asl_matrix_v1\qwen_runs"
set "EVALS=%REPO_ROOT%\artifacts\paper1\asl_matrix_v1\eval"

if not exist "%PYTHON%" (
    echo XPU Python is unavailable: %PYTHON% 1>&2
    exit /b 1
)

pushd "%REPO_ROOT%"
call :run q3s1_seed11 configs\paper1\asl_matrix_q3s1_capture_delta_xpu.json
if errorlevel 1 goto :failed
call :run q3s2_seed11 configs\paper1\asl_matrix_q3s2_separate_xpu.json
if errorlevel 1 goto :failed
call :run q3s3_seed11 configs\paper1\asl_matrix_q3s3_hybrid_xpu.json
if errorlevel 1 goto :failed
popd
exit /b 0

:run
set "NAME=%~1"
set "CONFIG=%~2"
set "RUN=%RUNS%\%NAME%"
set "EVALUATION=%EVALS%\%NAME%\autonomous"

if not exist "%RUN%\training_report.json" (
    "%PYTHON%" -u -m ccpu paper1 train-qwen-asl-patch --config "%CONFIG%" --train "%DATA%\train.jsonl" --dev "%DATA%\dev.jsonl" --output-dir "%RUN%"
    if errorlevel 1 exit /b 1
)

if not exist "%EVALUATION%\summary.json" (
    "%PYTHON%" -u -m ccpu paper1 evaluate-qwen-asl-patch --config "%CONFIG%" --state "%RUN%\trainable_patch_state.safetensors" --eval "%SOURCE%\test.jsonl" --train-split "%SOURCE%\train.jsonl" --output-dir "%EVALUATION%" --checkpoint-every 1
    if errorlevel 1 exit /b 1
)
exit /b 0

:failed
popd
exit /b 1
