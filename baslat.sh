#!/bin/bash

# Smarthome System Launcher (Git Bash / Linux)

echo "==================================================="
echo "  Smarthome Assistant - System Starting..."
echo "==================================================="
echo

ENV_PATH="$HOME/.conda/envs/smarthome"

# NanoMQ Docker Container Check
echo "[1/4] Checking NanoMQ Docker container..."
docker start nanomq >/dev/null 2>&1 || docker run -d -p 1883:1883 --name nanomq emqx/nanomq:latest >/dev/null 2>&1

# Launch services in separate CMD windows using environment binaries
echo "[2/4] Starting FastAPI server..."
cmd.exe /c start "FastAPI" cmd /k "\"$ENV_PATH/Scripts/uvicorn.exe\" app.main:app --port 8000"

echo "[3/4] Starting IoT Receiver Simulator..."
cmd.exe /c start "IoT Simulator" cmd /k "\"$ENV_PATH/python.exe\" iot_sim/iot_alici.py"

echo "[4/4] Starting Streamlit Dashboard..."
cmd.exe /c start "Streamlit Dashboard" cmd /k "\"$ENV_PATH/Scripts/streamlit.exe\" run app/dashboard.py"

echo
echo "==================================================="
echo "  All services started!"
echo "  - Streamlit: http://localhost:8501"
echo "  - FastAPI:   http://localhost:8000/docs"
echo "==================================================="
