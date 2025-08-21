from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it_inpo_stage_3/",
    repo_id="XuHuang/simpo_inpo_iter3_aug_20",
    repo_type="model",
)
