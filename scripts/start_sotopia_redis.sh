#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PORT=${SOTOPIA_REDIS_PORT:-6380}

REDIS_ROOT=${REDIS_ROOT:-"${REPO_ROOT}/.sotopia_bin/redis-stack-server-7.2.0-v10"}
REDIS_DATA_DIR=${REDIS_DATA_DIR:-"${REPO_ROOT}/.sotopia_redis/redis-data"}
SSL_LIB_DIR=${SSL_LIB_DIR:-"${REPO_ROOT}/.sotopia_bin/libssl1.1/usr/lib/x86_64-linux-gnu"}
REDIS_LOG=${REDIS_LOG:-"${REPO_ROOT}/.sotopia_redis/redis-${PORT}.log"}
REDIS_PID=${REDIS_PID:-"${REPO_ROOT}/.sotopia_redis/redis-${PORT}.pid"}

REDIS_SERVER="${REDIS_ROOT}/bin/redis-server"
REDIS_CLI="${REDIS_ROOT}/bin/redis-cli"
MODULE_DIR="${REDIS_ROOT}/lib"

if [[ ! -x "${REDIS_SERVER}" ]]; then
    echo "Missing Redis Stack server at ${REDIS_SERVER}."
    echo "Run: bash scripts/setup_sotopia_redis_stack.sh"
    exit 1
fi

if [[ ! -f "${SSL_LIB_DIR}/libssl.so.1.1" ]]; then
    echo "Missing local libssl1.1 at ${SSL_LIB_DIR}."
    echo "Run: bash scripts/setup_sotopia_redis_stack.sh"
    exit 1
fi

if [[ ! -s "${REDIS_DATA_DIR}/dump.rdb" ]]; then
    echo "Missing SOTOPIA Redis dump at ${REDIS_DATA_DIR}/dump.rdb."
    echo "Run: bash scripts/setup_sotopia_redis_stack.sh"
    exit 1
fi

mkdir -p "$(dirname "${REDIS_LOG}")" "${REDIS_DATA_DIR}"

export LD_LIBRARY_PATH="${SSL_LIB_DIR}:${MODULE_DIR}:${LD_LIBRARY_PATH:-}"

if [[ -f "${REDIS_PID}" ]] && kill -0 "$(cat "${REDIS_PID}")" 2>/dev/null; then
    echo "Redis Stack already running on port ${PORT}; PID $(cat "${REDIS_PID}")."
    echo "Use: export REDIS_OM_URL=redis://localhost:${PORT}"
    exit 0
fi

if "${REDIS_CLI}" -p "${PORT}" ping >/dev/null 2>&1; then
    echo "A Redis-compatible server is already responding on port ${PORT}."
    echo "Use: export REDIS_OM_URL=redis://localhost:${PORT}"
    exit 0
fi

"${REDIS_SERVER}" \
    --port "${PORT}" \
    --dir "${REDIS_DATA_DIR}" \
    --protected-mode no \
    --daemonize yes \
    --logfile "${REDIS_LOG}" \
    --pidfile "${REDIS_PID}" \
    --loadmodule "${MODULE_DIR}/rediscompat.so" \
    --loadmodule "${MODULE_DIR}/redisearch.so" MAXSEARCHRESULTS 10000 MAXAGGREGATERESULTS 10000 \
    --loadmodule "${MODULE_DIR}/redistimeseries.so" \
    --loadmodule "${MODULE_DIR}/rejson.so" \
    --loadmodule "${MODULE_DIR}/redisbloom.so" \
    --loadmodule "${MODULE_DIR}/redisgears.so" v8-plugin-path "${MODULE_DIR}/libredisgears_v8_plugin.so"

sleep 2
"${REDIS_CLI}" -p "${PORT}" ping
"${REDIS_CLI}" -p "${PORT}" dbsize

echo "Redis Stack log: ${REDIS_LOG}"
echo "Use: export REDIS_OM_URL=redis://localhost:${PORT}"
