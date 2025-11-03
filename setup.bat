@echo off
setlocal enabledelayedexpansion

REM ======================================
REM setup_ember.bat
REM Scarica dataset e chiede se usare Conda o Pip
REM ======================================

echo ==========================================
echo   Setup EMbER Installer
echo ==========================================
echo.

REM === CONFIGURAZIONE ===
set DEST_DIR=ember_datasets
set URL1=https://ember.elastic.co/ember_dataset.tar.bz2
set URL2=https://ember.elastic.co/ember_dataset_2017_2.tar.bz2
set URL3=https://ember.elastic.co/ember_dataset_2018_2.tar.bz2

echo === EMBER setup (Windows) ===

if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"
cd /d "%DEST_DIR%"

REM === DOWNLOAD FILE ===
call :download "%URL1%"
call :download "%URL2%"
call :download "%URL3%"

REM == DOMANDA ESTRAZIONE ===
set /p EXTRACTION=Do you want to extract the files now? (Y/N):
if /I "%EXTRACTION%"=="Y" goto :EXTR
if /I "%EXTRACTION%"=="y" goto :EXTR

:EXTR
REM === ESTRAZIONE ===
for %%F in (*.tar.bz2) do (
    if exist "%%F" (
        echo Extracting %%F ...
        where tar >nul 2>&1
        if !errorlevel! == 0 (
            tar -xjf "%%F"
        ) else (
            echo WARNING: tar non trovato. Installa tar o 7-zip per estrarre i file.
        )
    ) else (
        echo File %%F non trovato, salto.
    )
)

cd ..

REM === DOMANDA CONDA/PIP ===
set /p USE_CONDA=Do you want to install Ember using Conda? (Y/N): 

if /I "%USE_CONDA%"=="Y" goto CONDA
if /I "%USE_CONDA%"=="y" goto CONDA

goto PIP


:CONDA
echo [INFO] Modalita CONDA selezionata.

where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERRORE: conda non trovato nel PATH.
    pause
    exit /b 1
)

echo [STEP] Clonazione repository Ember...
git clone https://github.com/elastic/ember.git

echo [STEP] Aggiunta canale conda-forge...
conda config --add channels conda-forge

echo [STEP] Installazione pacchetti...
cd ember
conda install --file requirements_conda.txt -y

echo [STEP] Installazione Ember...
python -m pip install .

echo.
echo [OK] Installazione completata in modalita CONDA.
pause
exit /b 0


:PIP
echo [INFO] Modalita PIP selezionata (default).

where pip >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERRORE: pip non trovato nel PATH.
    pause
    exit /b 1
)

echo [STEP] Clonazione repository Ember...
git clone https://github.com/elastic/ember.git

echo [STEP] Installazione pacchetti...
pip install -r "%~dp0requirements.txt"


echo [STEP] Installazione Ember...
cd ember
python -m pip install .

echo.
echo [OK] Installazione completata in modalita PIP.
pause
exit /b 0


REM === SUBROUTINE PER DOWNLOAD ===
:download
set URL=%~1
for %%A in ("%URL%") do set FNAME=%%~nxA

if exist "%FNAME%" (
    echo %FNAME% gia' presente, salto download.
    goto :eof
)

where curl >nul 2>&1
if %ERRORLEVEL%==0 (
    echo Downloading %URL% con curl...
    curl -L -o "%FNAME%" "%URL%"
    goto :eof
)

echo curl non trovato, uso PowerShell...
powershell -Command "try { Invoke-WebRequest -Uri '%URL%' -OutFile '%FNAME%' -UseBasicParsing } catch { exit 1 }"
if %ERRORLEVEL% NEQ 0 (
    echo ERRORE: download fallito per %URL%.
)
goto :eof
