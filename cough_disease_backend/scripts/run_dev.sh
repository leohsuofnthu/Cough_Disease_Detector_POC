#!/usr/bin/env bash

set -euo pipefail

export BACKEND_LOG_LEVEL=${BACKEND_LOG_LEVEL:-INFO}

uvicorn app.main:app --reload --port "${PORT:-8000}" --host "0.0.0.0"

