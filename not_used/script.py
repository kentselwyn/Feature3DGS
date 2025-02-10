import torch
import subprocess

for scene_name in ['pgt_7scenes_chess', 'pgt_7scenes_pumpkin', 'pgt_7scenes_redkitchen']:
    command = ['bash', 'zenith_scripts/loc_inference.sh', scene_name]
    subprocess.run(command, check=True)
    torch.cuda.empty_cache()

# python script.py
