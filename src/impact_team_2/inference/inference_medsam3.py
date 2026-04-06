from huggingface_hub import snapshot_download
from impact_team_2.vendor.medsam3.lora_layers import LoRAConfig, apply_lora_to_model


lora_path = snapshot_download(repo_id="lal-Joey/MedSAM3_v1")

print(lora_path)



