from impact_team_2.train.data import SAM3Dataset
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Explicitly load the .env.local file
load_dotenv(dotenv_path=f"{os.path.dirname(__file__)}/../../../.env.local")

hf_token = os.getenv("HF_TOKEN")
login(token=hf_token, add_to_git_credential=True)