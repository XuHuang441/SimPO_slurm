from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it_op_tdpo_stage_3/",
    repo_id="XuHuang/simpo_inpo_op_iter3_aug22",
    repo_type="model",
)
