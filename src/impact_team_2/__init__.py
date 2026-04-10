from pathlib import Path
from dotenv import load_dotenv

# Explicitly load the .env.local file relative to the repo root.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env.local")
