#!/usr/bin/env bash
# Create/update the impact-team-2 conda env and (if sourced) activate it.
#
# Usage:
#   source setup.sh        # creates/updates env, activates it in this shell
#   ./setup.sh             # creates/updates env, prints activation hint
#
# Re-running is safe: an existing env is updated in place.
#
# Platform behavior:
#   - Linux + NVIDIA: uses environment.yml (torch 2.7/cu126 + TF 2.19 stack).
#   - Apple Silicon: creates a Python-only env, then pip-installs MPS torch
#     plus tensorflow-macos / tensorflow-metal.
#   - Linux without nvidia-smi: creates a Python-only env, then pip-installs
#     CPU torch + tensorflow-cpu.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ENV_FILE="${REPO_ROOT}/environment.yml"
ENV_NAME="${CONDA_ENV_NAME:-impact-team-2}"

if ! command -v conda >/dev/null 2>&1; then
    echo "[setup] conda not found on PATH — install Miniconda/Anaconda first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

env_exists() {
    conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"
}

create_or_update_from_yml() {
    cd "${REPO_ROOT}"
    if env_exists; then
        echo "[setup] updating '${ENV_NAME}' from ${ENV_FILE}"
        conda env update --name "${ENV_NAME}" --file "${ENV_FILE}" --prune
    else
        echo "[setup] creating '${ENV_NAME}' from ${ENV_FILE}"
        conda env create --name "${ENV_NAME}" --file "${ENV_FILE}"
    fi
}

create_python_only_env() {
    if env_exists; then
        echo "[setup] reusing existing '${ENV_NAME}'"
    else
        echo "[setup] creating '${ENV_NAME}' with python=3.12"
        conda create --yes --name "${ENV_NAME}" --channel conda-forge python=3.12 pip
    fi
}

pip_in_env() {
    conda run --no-capture-output --name "${ENV_NAME}" python -m pip "$@"
}

if [ "${OS_NAME}" = "Darwin" ] && [ "${ARCH_NAME}" = "arm64" ]; then
    echo "[setup] Apple Silicon — MPS torch + tensorflow-macos/metal"
    create_python_only_env
    pip_in_env install --upgrade pip
    pip_in_env install "torch>=2.7,<2.8" "torchvision>=0.22,<0.23"
    pip_in_env install "tensorflow-macos==2.16.*" "tensorflow-metal"
    pip_in_env install -e "${REPO_ROOT}[unet]"
elif command -v nvidia-smi >/dev/null 2>&1; then
    # NVIDIA drivers are forward-compatible across CUDA minors; the working
    # stack is cu126 for all 12.x drivers, cu118 for older drivers.
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -1 | awk '{print $3}')
    case "${CUDA_VER}" in
        12.*)        TORCH_INDEX="cu126" ;;
        11.8*|11.9*) TORCH_INDEX="cu118" ;;
        *)           TORCH_INDEX="" ;;
    esac
    if [ "${TORCH_INDEX}" = "cu126" ]; then
        echo "[setup] driver CUDA ${CUDA_VER} → using ${ENV_FILE} (cu126 + TF 2.19)"
        create_or_update_from_yml
    else
        echo "[setup] driver CUDA ${CUDA_VER:-unknown} unmapped — creating env manually"
        create_python_only_env
        pip_in_env install --upgrade pip
        if [ -n "${TORCH_INDEX}" ]; then
            pip_in_env install --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" \
                "torch>=2.7,<2.8" "torchvision>=0.22,<0.23"
        else
            pip_in_env install --index-url "https://download.pytorch.org/whl/cpu" \
                "torch>=2.7,<2.8" "torchvision>=0.22,<0.23"
        fi
        pip_in_env install "tensorflow==2.19.*"
        pip_in_env install -e "${REPO_ROOT}[unet]"
    fi
else
    echo "[setup] no nvidia-smi — CPU torch + tensorflow-cpu"
    create_python_only_env
    pip_in_env install --upgrade pip
    pip_in_env install --index-url "https://download.pytorch.org/whl/cpu" \
        "torch>=2.7,<2.8" "torchvision>=0.22,<0.23"
    pip_in_env install "tensorflow-cpu==2.19.*"
    pip_in_env install -e "${REPO_ROOT}[unet]"
fi

if [ ! -f "${REPO_ROOT}/.env.local" ] && [ -f "${REPO_ROOT}/.env.example" ]; then
    cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env.local"
    echo "[setup] created .env.local from .env.example — set HF_TOKEN before running"
fi

# Install an activate.d hook that adds the pip-installed NVIDIA wheel libs to
# LD_LIBRARY_PATH and points XLA at libdevice.10.bc.
CONDA_PREFIX_DIR="$(conda run --name "${ENV_NAME}" printenv CONDA_PREFIX 2>/dev/null || true)"
if [ -n "${CONDA_PREFIX_DIR}" ] && [ -d "${CONDA_PREFIX_DIR}" ]; then
    mkdir -p "${CONDA_PREFIX_DIR}/etc/conda/activate.d" "${CONDA_PREFIX_DIR}/etc/conda/deactivate.d"
    cat > "${CONDA_PREFIX_DIR}/etc/conda/activate.d/impact_team_2.sh" <<'ACT'
# impact_team_2: add pip-installed NVIDIA wheel libs (cuDNN/cuBLAS/NCCL) to the
# loader path, and point XLA at libdevice.10.bc from nvidia-cuda-nvcc-cu12.
export _IT2_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export _IT2_OLD_XLA_FLAGS="${XLA_FLAGS:-}"
_IT2_LIBS=""
for _d in "${CONDA_PREFIX}"/lib/python*/site-packages/nvidia/*/lib; do
    [ -d "${_d}" ] && _IT2_LIBS="${_IT2_LIBS:+${_IT2_LIBS}:}${_d}"
done
if [ -n "${_IT2_LIBS}" ]; then
    export LD_LIBRARY_PATH="${_IT2_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
for _d in "${CONDA_PREFIX}"/lib/python*/site-packages/nvidia/cuda_nvcc; do
    [ -d "${_d}" ] && export XLA_FLAGS="--xla_gpu_cuda_data_dir=${_d}"
done
unset _IT2_LIBS _d
ACT
    cat > "${CONDA_PREFIX_DIR}/etc/conda/deactivate.d/impact_team_2.sh" <<'DEA'
if [ -n "${_IT2_OLD_LD_LIBRARY_PATH+x}" ]; then
    export LD_LIBRARY_PATH="${_IT2_OLD_LD_LIBRARY_PATH}"
    unset _IT2_OLD_LD_LIBRARY_PATH
fi
if [ -n "${_IT2_OLD_XLA_FLAGS+x}" ]; then
    if [ -z "${_IT2_OLD_XLA_FLAGS}" ]; then
        unset XLA_FLAGS
    else
        export XLA_FLAGS="${_IT2_OLD_XLA_FLAGS}"
    fi
    unset _IT2_OLD_XLA_FLAGS
fi
DEA
    echo "[setup] installed activate.d/deactivate.d hooks for NVIDIA wheel libs + XLA_FLAGS"
fi

# CUDA / TF availability check — non-fatal.
conda run --no-capture-output --name "${ENV_NAME}" python - <<'PY'
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
    print(f"[setup] torch CUDA: available ({n} device(s): {names})")
else:
    print("[setup] torch CUDA: NOT available — training will be unusably slow on CPU")

try:
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
    conda activate "${ENV_NAME}"
    echo "[setup] conda env '${ENV_NAME}' activated"
else
    echo "[setup] done. Activate with: conda activate ${ENV_NAME}"
fi
