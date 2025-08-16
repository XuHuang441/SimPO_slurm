from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter2",
    repo_id="XuHuang/inpo_iter1",
    repo_type="dataset"
)
