import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(dotenv_path=_repo_root / ".env.local")

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise EnvironmentError("HF_TOKEN not set. Add it to .env.local or export it in your shell.")
login(token=hf_token, add_to_git_credential=True)


def __getattr__(name):
    if name in ("predict", "build_predictor"):
        from . import _inference_sam3 as _m
        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["predict", "build_predictor"]
