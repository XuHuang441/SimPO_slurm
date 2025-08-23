from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it-simpo-iter1_on_policy/",
    repo_id="XuHuang/simpo_onPolicy_iter1_aug23",
    repo_type="model",
)
