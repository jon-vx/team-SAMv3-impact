import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(dotenv_path=_repo_root / ".env.local")

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise EnvironmentError(
        "HF_TOKEN is not set. Create a .env.local file in the repo root with:\n"
        "  HF_TOKEN=hf_yourtoken\n"
        "or export HF_TOKEN in your shell."
    )
login(token=hf_token, add_to_git_credential=True)

from ._inference_medsam3 import predict, build_predictor
__all__ = ["predict", "build_predictor"]

