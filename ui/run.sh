#!/usr/bin/env bash
# ==============================================================================
# COLESION-AI // HUD & Data Explorer Launcher
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Initialize ui/.env from .env.example if missing
if [[ ! -f "${SCRIPT_DIR}/.env" ]]; then
  if [[ -f "${SCRIPT_DIR}/.env.example" ]]; then
    echo "[INFO] Creating ui/.env from .env.example..."
    cp "${SCRIPT_DIR}/.env.example" "${SCRIPT_DIR}/.env"
  else
    echo "[WARN] No .env or .env.example found. Using defaults."
  fi
fi

# 2. Resolve Python environment: prefer repo .venv, fallback to uv run python, then python3
if [[ -f "${SCRIPT_DIR}/../.venv/bin/python" ]]; then
  PYTHON="${SCRIPT_DIR}/../.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  PYTHON="uv run python"
else
  PYTHON="python3"
fi

# 3. Read PORT from ui/.env if present, otherwise default to 8000
PORT=8000
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  ENV_PORT=$(grep -E '^PORT=' "${SCRIPT_DIR}/.env" | cut -d '=' -f2 | tr -d ' "' | tr -d '\r')
  if [[ -n "${ENV_PORT}" ]]; then
    PORT="${ENV_PORT}"
  fi
fi
URL="http://localhost:${PORT}/"

# 4. Check if calibration state exists or if --recalibrate / -r was passed
RECALIBRATE=false
if [[ "$1" == "--recalibrate" || "$1" == "-r" ]]; then
  RECALIBRATE=true
fi

STATE_PARQUET="${SCRIPT_DIR}/state/calibrated_cases.parquet"
STATE_METRICS="${SCRIPT_DIR}/state/metrics.json"

if [[ "${RECALIBRATE}" == "true" || ! -f "${STATE_PARQUET}" || ! -f "${STATE_METRICS}" ]]; then
  echo "[INFO] Building calibration state (Temperature Scaling, OOD Threshold, 6-Twins Index)..."
  $PYTHON "${SCRIPT_DIR}/calibrate.py"
  echo "[INFO] Calibration state successfully built."
fi

# 5. Free port if occupied
OCCUPYING_PID=$(lsof -ti :${PORT} 2>/dev/null || true)
if [[ -n "${OCCUPYING_PID}" ]]; then
  echo "[WARN] Port ${PORT} occupied by PID ${OCCUPYING_PID}. Terminating..."
  kill -9 ${OCCUPYING_PID} 2>/dev/null || true
  sleep 0.5
fi

# 6. Open browser on macOS / Linux
if command -v open >/dev/null 2>&1; then
  (sleep 0.8 && open "${URL}") &
elif command -v xdg-open >/dev/null 2>&1; then
  (sleep 0.8 && xdg-open "${URL}") &
fi

echo "[INFO] Starting COLESION-AI Tactical HUD on ${URL}..."
echo "[INFO] Using Python: ${PYTHON}"
cd "${SCRIPT_DIR}"
exec $PYTHON "${SCRIPT_DIR}/server.py"
