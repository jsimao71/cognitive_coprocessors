@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "TECTONIC_VERSION=0.17.0"
set "SOURCE_DATE_EPOCH=1787875200"
set "TECTONIC_EXE=%TECTONIC%"
if not defined TECTONIC_EXE (
    where tectonic >nul 2>nul
    if not errorlevel 1 set "TECTONIC_EXE=tectonic"
)
if not defined TECTONIC_EXE (
    set "LOCAL_TECTONIC=%LOCALAPPDATA%\ccpu-tools\tectonic-%TECTONIC_VERSION%\tectonic.exe"
    if exist "!LOCAL_TECTONIC!" set "TECTONIC_EXE=!LOCAL_TECTONIC!"
)

if not defined TECTONIC_EXE (
    echo Tectonic %TECTONIC_VERSION% is required. Set TECTONIC to tectonic.exe. 1>&2
    exit /b 1
)
if not "%TECTONIC_EXE%"=="tectonic" if not exist "%TECTONIC_EXE%" (
    echo Tectonic executable not found: %TECTONIC_EXE% 1>&2
    exit /b 1
)

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
set /a PAPER_COUNT=0

for /r "%REPO_ROOT%\docs\papers" %%F in (paper*.tex) do (
    set /a PAPER_COUNT+=1
    echo Building %%F
    pushd "%%~dpF"
    "%TECTONIC_EXE%" "%%~nxF"
    if errorlevel 1 (
        popd
        echo Tectonic failed for %%F. 1>&2
        exit /b 1
    )
    popd
    if not exist "%%~dpnF.pdf" (
        echo Expected PDF was not created: %%~dpnF.pdf 1>&2
        exit /b 1
    )
)

if %PAPER_COUNT% equ 0 (
    echo No paper sources found under %REPO_ROOT%\docs\papers. 1>&2
    exit /b 1
)

echo Built %PAPER_COUNT% paper PDFs with Tectonic %TECTONIC_VERSION%.
