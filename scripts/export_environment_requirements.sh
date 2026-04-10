#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_PATH="${1:-environment/current_environment_requirements.txt}"
OUTPUT_DIR="$(dirname "${OUTPUT_PATH}")"
mkdir -p "${OUTPUT_DIR}"

PYTHON_EXECUTABLE="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(sys.version.replace("\n", " "))')"
PYTHON_PLATFORM="$("${PYTHON_BIN}" -c 'import platform; print(platform.platform())')"
PIP_VERSION="$("${PYTHON_BIN}" -m pip --version)"
GENERATED_AT_UTC="$("${PYTHON_BIN}" - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"

{
  echo "# Exported from the currently active Python environment"
  echo "# generated_at_utc=${GENERATED_AT_UTC}"
  echo "# python_executable=${PYTHON_EXECUTABLE}"
  echo "# python_version=${PYTHON_VERSION}"
  echo "# platform=${PYTHON_PLATFORM}"
  echo "# pip=${PIP_VERSION}"
  echo
  "${PYTHON_BIN}" -m pip freeze --all | LC_ALL=C sort
} > "${OUTPUT_PATH}"

echo "Wrote ${OUTPUT_PATH}"
