@echo off
title Smarthome System Launcher

echo ===================================================
echo   Smarthome Assistant - System Starting...
echo ===================================================
echo.

:: Conda activate.bat bulma mantığı
set "CONDA_ACT="
if exist "%UserProfile%\anaconda3\Scripts\activate.bat" set "CONDA_ACT=%UserProfile%\anaconda3\Scripts\activate.bat"
if exist "%UserProfile%\miniconda3\Scripts\activate.bat" set "CONDA_ACT=%UserProfile%\miniconda3\Scripts\activate.bat"
if exist "%UserProfile%\AppData\Local\anaconda3\Scripts\activate.bat" set "CONDA_ACT=%UserProfile%\AppData\Local\anaconda3\Scripts\activate.bat"
if exist "%UserProfile%\AppData\Local\miniconda3\Scripts\activate.bat" set "CONDA_ACT=%UserProfile%\AppData\Local\miniconda3\Scripts\activate.bat"
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" set "CONDA_ACT=C:\ProgramData\anaconda3\Scripts\activate.bat"
if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" set "CONDA_ACT=C:\ProgramData\miniconda3\Scripts\activate.bat"

set "ENV_NAME=smarthome"
set "ENV_PATH=%UserProfile%\.conda\envs\smarthome"

if defined CONDA_ACT (
    echo Conda bulundu: %CONDA_ACT%
    call "%CONDA_ACT%" %ENV_NAME%
    set "ACT_CMD=call "%CONDA_ACT%" %ENV_NAME%"
) else (
    echo Conda PATH icinde aranıyor...
    call conda activate %ENV_NAME% 2>nul
    set "ACT_CMD=call conda activate %ENV_NAME%"
)

REM 1. NanoMQ Docker Container Check
echo [1/4] Checking NanoMQ Docker container...
docker start nanomq >nul 2>&1
if %errorlevel% neq 0 (
    docker run -d -p 1883:1883 --name nanomq emqx/nanomq:latest >nul 2>&1
)

REM 2. Launch FastAPI in new window
echo [2/4] Starting FastAPI server on port 8000...
if defined CONDA_ACT (
    start "FastAPI" cmd /k "%ACT_CMD% && uvicorn app.main:app --port 8000"
) else (
    start "FastAPI" cmd /k ""%ENV_PATH%\Scripts\uvicorn.exe" app.main:app --port 8000"
)

REM 3. Launch IoT Receiver in new window
echo [3/4] Starting IoT Receiver Simulator...
if defined CONDA_ACT (
    start "IoT Simulator" cmd /k "%ACT_CMD% && python iot_sim/iot_alici.py"
) else (
    start "IoT Simulator" cmd /k ""%ENV_PATH%\python.exe" iot_sim/iot_alici.py"
)

REM 4. Launch Streamlit Dashboard in new window
echo [4/4] Starting Streamlit Dashboard...
if defined CONDA_ACT (
    start "Streamlit Dashboard" cmd /k "%ACT_CMD% && streamlit run app/dashboard.py"
) else (
    start "Streamlit Dashboard" cmd /k ""%ENV_PATH%\Scripts\streamlit.exe" run app/dashboard.py"
)

echo.
echo ===================================================
echo   All services started in separate windows!
echo   - Streamlit: http://localhost:8501
echo   - FastAPI:   http://localhost:8000/docs
echo ===================================================
pause
