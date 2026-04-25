#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PORT=${SOTOPIA_REDIS_PORT:-6380}
REDIS_ROOT=${REDIS_ROOT:-"${REPO_ROOT}/.sotopia_bin/redis-stack-server-7.2.0-v10"}
SSL_LIB_DIR=${SSL_LIB_DIR:-"${REPO_ROOT}/.sotopia_bin/libssl1.1/usr/lib/x86_64-linux-gnu"}
REDIS_PID=${REDIS_PID:-"${REPO_ROOT}/.sotopia_redis/redis-${PORT}.pid"}
REDIS_CLI="${REDIS_ROOT}/bin/redis-cli"

export LD_LIBRARY_PATH="${SSL_LIB_DIR}:${REDIS_ROOT}/lib:${LD_LIBRARY_PATH:-}"

if [[ -x "${REDIS_CLI}" ]] && "${REDIS_CLI}" -p "${PORT}" ping >/dev/null 2>&1; then
    "${REDIS_CLI}" -p "${PORT}" shutdown nosave || true
    echo "Stopped Redis Stack on port ${PORT}."
    exit 0
fi

if [[ -f "${REDIS_PID}" ]] && kill -0 "$(cat "${REDIS_PID}")" 2>/dev/null; then
    kill "$(cat "${REDIS_PID}")"
    echo "Stopped Redis Stack PID $(cat "${REDIS_PID}")."
    exit 0
fi

echo "No Redis Stack process found for port ${PORT}."
