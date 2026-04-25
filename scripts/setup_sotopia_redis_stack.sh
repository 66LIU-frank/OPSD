#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

BIN_DIR=${SOTOPIA_BIN_DIR:-"${REPO_ROOT}/.sotopia_bin"}
REDIS_ROOT="${BIN_DIR}/redis-stack-server-7.2.0-v10"
REDIS_TARBALL="${BIN_DIR}/redis-stack-server.tar.gz"
LIBSSL_DEB="${BIN_DIR}/libssl1.1.deb"
LIBSSL_DIR="${BIN_DIR}/libssl1.1"
REDIS_DATA_DIR=${REDIS_DATA_DIR:-"${REPO_ROOT}/.sotopia_redis/redis-data"}

REDIS_TARBALL_URL=${REDIS_TARBALL_URL:-https://packages.redis.io/redis-stack/redis-stack-server-7.2.0-v10.focal.x86_64.tar.gz}
LIBSSL_URL=${LIBSSL_URL:-http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb}

HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
HF_ENDPOINT=${HF_ENDPOINT%/}
SOTOPIA_PI_BASE_URL=${SOTOPIA_PI_BASE_URL:-"${HF_ENDPOINT}/datasets/cmu-lti/sotopia-pi/resolve/main"}
DUMP_URL=${DUMP_URL:-"${SOTOPIA_PI_BASE_URL}/dump.rdb?download=true"}

mkdir -p "${BIN_DIR}" "${REDIS_DATA_DIR}"

if [[ ! -s "${REDIS_TARBALL}" ]]; then
    echo "Downloading Redis Stack:"
    echo "${REDIS_TARBALL_URL}"
    curl -L --fail --retry 3 -o "${REDIS_TARBALL}" "${REDIS_TARBALL_URL}"
fi

if [[ ! -x "${REDIS_ROOT}/bin/redis-server" ]]; then
    tar -xzf "${REDIS_TARBALL}" -C "${BIN_DIR}"
fi

if [[ ! -s "${LIBSSL_DEB}" ]]; then
    echo "Downloading libssl1.1:"
    echo "${LIBSSL_URL}"
    curl -L --fail --retry 3 -o "${LIBSSL_DEB}" "${LIBSSL_URL}"
fi

if [[ ! -f "${LIBSSL_DIR}/usr/lib/x86_64-linux-gnu/libssl.so.1.1" ]]; then
    mkdir -p "${LIBSSL_DIR}"
    dpkg-deb -x "${LIBSSL_DEB}" "${LIBSSL_DIR}"
fi

if [[ ! -s "${REDIS_DATA_DIR}/dump.rdb" ]]; then
    echo "Downloading SOTOPIA-pi Redis dump:"
    echo "${DUMP_URL}"
    curl -L --fail --retry 3 -o "${REDIS_DATA_DIR}/dump.rdb" "${DUMP_URL}"
fi

bash "${REPO_ROOT}/scripts/start_sotopia_redis.sh"
