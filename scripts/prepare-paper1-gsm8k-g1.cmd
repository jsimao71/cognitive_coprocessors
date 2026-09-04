@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
set "PYTHON=%USERPROFILE%\.venvs\modal-llm-xpu\Scripts\python.exe"
set "PYTHONPATH=%REPO_ROOT%\src"
set "STRICT=%REPO_ROOT%\artifacts\paper1\dsl\openrouter_full_v1\recovery_v2\consolidated_run3\combined_strict.jsonl"
set "SOURCE=%REPO_ROOT%\artifacts\paper1\dsl\openrouter_full_v1\inputs\gsm8k_full.jsonl"
set "FROZEN=%REPO_ROOT%\artifacts\paper1\asl_matrix_v1\data\source"
set "OUTPUT=%REPO_ROOT%\artifacts\paper1\gsm8k_scale_v1\g1_f0_4500"

if not exist "%PYTHON%" (
    echo XPU Python is unavailable: %PYTHON% 1>&2
    exit /b 1
)
if not exist "%STRICT%" (
    echo Final strict OpenRouter corpus is unavailable: %STRICT% 1>&2
    exit /b 1
)

pushd "%REPO_ROOT%"
"%PYTHON%" -m ccpu.paper1.e3 prepare-gsm8k ^
    --strict "%STRICT%" ^
    --source "%SOURCE%" ^
    --frozen-data-dir "%FROZEN%" ^
    --output-dir "%OUTPUT%" ^
    --target 4500 ^
    --epochs 10 ^
    --seed 11
if errorlevel 1 goto :failed
popd
exit /b 0

:failed
popd
exit /b 1
