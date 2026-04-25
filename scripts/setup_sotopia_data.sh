#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/lsg/miniconda3/envs/opsd/bin/python}
DATA_DIR=${SOTOPIA_DATA_DIR:-"${REPO_ROOT}/.sotopia_data"}
RAW_DIR=${RAW_DIR:-"${DATA_DIR}/raw"}
RC_OPD_DIR=${RC_OPD_DIR:-"${DATA_DIR}/rc_opd"}
EPISODES_FILE=${EPISODES_FILE:-"${RAW_DIR}/sotopia_pi_episodes.jsonl"}
OUTPUT_FILE=${OUTPUT_FILE:-"${RC_OPD_DIR}/sotopia_pi_rc_opd.jsonl"}
MAX_EXAMPLES=${MAX_EXAMPLES:-20000}

HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
HF_ENDPOINT=${HF_ENDPOINT%/}
SOTOPIA_PI_BASE_URL=${SOTOPIA_PI_BASE_URL:-"${HF_ENDPOINT}/datasets/cmu-lti/sotopia-pi/resolve/main"}
EPISODES_URL=${EPISODES_URL:-"${SOTOPIA_PI_BASE_URL}/sotopia_pi_episodes.jsonl?download=true"}

mkdir -p "${RAW_DIR}" "${RC_OPD_DIR}"

if [[ ! -s "${EPISODES_FILE}" ]]; then
    echo "Downloading SOTOPIA-pi episodes:"
    echo "${EPISODES_URL}"
    curl -L --fail --retry 3 -o "${EPISODES_FILE}" "${EPISODES_URL}"
else
    echo "Using existing episodes file: ${EPISODES_FILE}"
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/export_sotopia_rc_opd.py" \
    --episodes-jsonl "${EPISODES_FILE}" \
    --output-file "${OUTPUT_FILE}" \
    --max-examples "${MAX_EXAMPLES}" \
    --require-dialogue

echo "RC-OPD dataset written to: ${OUTPUT_FILE}"
