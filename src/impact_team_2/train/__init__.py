from impact_team_2.train.data import SAM3Dataset
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Explicitly load the .env.local file
load_dotenv(dotenv_path=f"{os.path.dirname(__file__)}/../../../.env.local")

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token, add_to_git_credential=True)
else:
    import warnings
    warnings.warn(
        "HF_TOKEN not found in .env.local — skipping Hugging Face login. "
        "Training requires a valid token to download model weights. "
        "Set HF_TOKEN in .env.local (see .env.example).",
        stacklevel=2,
    )

from impact_team_2.train.med_sam import (
    train_medsam3,
    export_coco_dataset,
    LoRAConfig,
    TrainingConfig,
)

__all__ = [
    "SAM3Dataset",
    "train_medsam3",
    "export_coco_dataset",
    "LoRAConfig",
    "TrainingConfig",
]