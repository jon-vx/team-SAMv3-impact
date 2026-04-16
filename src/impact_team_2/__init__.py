import os

# TF must be told to grow its arena on demand before it's imported, else it
# grabs the whole GPU and OOMs torch when both run on the same device.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env.local")

from impact_team_2.api import predict, evaluate, clear_cache

__all__ = ["predict", "evaluate", "clear_cache"]
