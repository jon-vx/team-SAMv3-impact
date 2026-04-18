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
    # NVIDIA drivers are forward-compatible across CUDA minors, so we standardize
    # on cu126 for all 12.x drivers (torch 2.7.x ships this wheel). Older
    # drivers (11.8/11.9) get cu118. Anything unmapped falls back to CPU.
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -1 | awk '{print $3}')
    case "${CUDA_VER}" in
        12.*) TORCH_INDEX="cu126" ;;
        11.8*|11.9*) TORCH_INDEX="cu118" ;;
        *)  echo "[setup] driver CUDA ${CUDA_VER:-unknown} unmapped — falling back to CPU torch"
            TORCH_INDEX="cpu" ;;
    esac
    TF_PIN="tensorflow==2.19.*"
    if [ -n "${TORCH_INDEX}" ]; then
        echo "[setup] driver CUDA ${CUDA_VER} → torch=${TORCH_INDEX}, tf=${TF_PIN}"
    fi
else
    TORCH_INDEX="cpu"
    TF_PIN="tensorflow-cpu==2.19.*"
    echo "[setup] no nvidia-smi found → installing CPU torch + CPU tensorflow"
fi

# Install torch from the matched index first so its CUDA-specific wheels are
# already in place before the editable install runs.
if [ -n "${TORCH_INDEX}" ]; then
    "${PIP}" install --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" torch torchvision
fi

"${PIP}" install -e "${REPO_ROOT}[unet]" "${TF_PIN}" ${TF_EXTRA:+${TF_EXTRA}}

if [ ! -f "${REPO_ROOT}/.env.local" ] && [ -f "${REPO_ROOT}/.env.example" ]; then
    cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env.local"
    echo "[setup] created .env.local from .env.example — set HF_TOKEN before running"
fi

ACTIVATE="${VENV_DIR}/bin/activate"
# Remove any previous (possibly broken) patch so re-runs always get the current version.
if grep -q "impact_team_2: nvidia wheel libs" "${ACTIVATE}" 2>/dev/null; then
    # Strip from the marker line to EOF.
    sed -i '/# impact_team_2: nvidia wheel libs/,$d' "${ACTIVATE}"
fi
cat >> "${ACTIVATE}" <<'ACT'

# impact_team_2: nvidia wheel libs -- add cuDNN/cuBLAS/NCCL from pip wheels to
# loader path, and point XLA at libdevice.10.bc from the nvidia-cuda-nvcc-cu12
# wheel so TF/UNet training works on GPU.
_IT2_VENV="${VIRTUAL_ENV:-}"
if [ -n "${_IT2_VENV}" ]; then
    _IT2_LIBS=""
    for _d in "${_IT2_VENV}"/lib/python*/site-packages/nvidia/*/lib; do
        [ -d "${_d}" ] && _IT2_LIBS="${_IT2_LIBS:+${_IT2_LIBS}:}${_d}"
    done
    if [ -n "${_IT2_LIBS}" ]; then
        export LD_LIBRARY_PATH="${_IT2_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    for _d in "${_IT2_VENV}"/lib/python*/site-packages/nvidia/cuda_nvcc; do
        [ -d "${_d}" ] && export XLA_FLAGS="--xla_gpu_cuda_data_dir=${_d}"
    done
    unset _IT2_LIBS _d
fi
unset _IT2_VENV
ACT
echo "[setup] patched venv activate script to add NVIDIA wheel libs to LD_LIBRARY_PATH"

source "${ACTIVATE}"

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

if (return 0 2>/dev/null); then
    echo "[setup] venv activated"
else
    echo "[setup] done. Activate with: source ${VENV_DIR}/bin/activate"
fi
