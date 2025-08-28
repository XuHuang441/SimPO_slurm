from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it_inpo_stage_2/",
    repo_id="XuHuang/inpo_iter2_aug28",
    repo_type="model",
)
