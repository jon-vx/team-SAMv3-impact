#!/usr/bin/env bash
# Create a venv, install impact_team_2 + dependencies, and (if sourced) activate it.
#
# Usage:
#   source setup.sh        # creates/activates venv in current shell
#   ./setup.sh             # creates venv, installs deps, then prints activation hint
#
# Re-running is safe: an existing venv is reused.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
VENV_DIR="${REPO_ROOT}/venv"
PYTHON="${PYTHON:-python3.12}"

if [ ! -d "${VENV_DIR}" ]; then
    echo "[setup] creating venv at ${VENV_DIR} (using ${PYTHON})"
    "${PYTHON}" -m venv "${VENV_DIR}"
else
    echo "[setup] reusing existing venv at ${VENV_DIR}"
fi

# Use the venv's pip directly so this works whether or not we're sourced.
PIP="${VENV_DIR}/bin/pip"
"${PIP}" install --upgrade pip

# Detect CUDA version from the NVIDIA driver and install the matching PyTorch
# wheel BEFORE the editable install, so the project's torch dep is already
# satisfied by the right CUDA build. Falls back to the CPU wheel if no GPU.
TORCH_INDEX=""
TF_PIN=""
TF_EXTRA=""

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

if [ "${OS_NAME}" = "Darwin" ] && [ "${ARCH_NAME}" = "arm64" ]; then
    echo "[setup] detected Apple Silicon (${OS_NAME}/${ARCH_NAME}) → MPS torch + tensorflow-macos/metal"
    # Default PyPI torch wheel on macOS arm64 already supports MPS — no index override.
    TF_PIN="tensorflow-macos==2.16.*"
    TF_EXTRA="tensorflow-metal"
elif command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -1 | awk '{print $3}')
    case "${CUDA_VER}" in
        12.8*|12.9*) TORCH_INDEX="cu128"; TF_PIN="tensorflow[and-cuda]==2.17.*" ;;
        12.6*|12.7*) TORCH_INDEX="cu126"; TF_PIN="tensorflow[and-cuda]==2.17.*" ;;
        12.4*|12.5*) TORCH_INDEX="cu124"; TF_PIN="tensorflow[and-cuda]==2.17.*" ;;
        12.1*|12.2*|12.3*) TORCH_INDEX="cu121"; TF_PIN="tensorflow[and-cuda]==2.16.*" ;;
        11.8*|11.9*|12.0*) TORCH_INDEX="cu118"; TF_PIN="tensorflow==2.14.*" ;;
        "") echo "[setup] nvidia-smi present but couldn't parse CUDA version — falling back to CPU TF"
            TF_PIN="tensorflow-cpu==2.17.*" ;;
        *)  echo "[setup] driver CUDA ${CUDA_VER} has no matching wheel — falling back to CPU TF"
            TF_PIN="tensorflow-cpu==2.17.*" ;;
    esac
    if [ -n "${TORCH_INDEX}" ]; then
        echo "[setup] driver CUDA ${CUDA_VER} → torch=${TORCH_INDEX}, tf=${TF_PIN}"
    fi
else
    TORCH_INDEX="cpu"
    TF_PIN="tensorflow-cpu==2.17.*"
    echo "[setup] no nvidia-smi found → installing CPU torch + CPU tensorflow"
fi

if [ -n "${TORCH_INDEX}" ]; then
    "${PIP}" install --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" torch torchvision
fi

"${PIP}" install -e "${REPO_ROOT}[unet]" "${TF_PIN}" ${TF_EXTRA:+${TF_EXTRA}}

if [ ! -f "${REPO_ROOT}/.env.local" ] && [ -f "${REPO_ROOT}/.env.example" ]; then
    cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env.local"
    echo "[setup] created .env.local from .env.example — set HF_TOKEN before running"
fi

# CUDA availability check — non-fatal, just a heads up.
"${VENV_DIR}/bin/python" - <<'PY'
import torch
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
    print(f"[setup] torch CUDA: available ({n} device(s): {names})")
else:
    print("[setup] torch CUDA: NOT available — training will be unusably slow on CPU")

try:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[setup] TF GPUs: {len(gpus)} ({', '.join(g.name for g in gpus)})")
    else:
        print("[setup] TF GPUs: none visible — UNet will run on CPU")
except ImportError:
    print("[setup] TF not installed (unet extra missing)")
except Exception as e:
    print(f"[setup] TF import failed: {e}")
PY

# Activate iff this script was sourced; otherwise tell the user how.
if (return 0 2>/dev/null); then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    echo "[setup] venv activated"
else
    echo "[setup] done. Activate with: source ${VENV_DIR}/bin/activate"
fi
