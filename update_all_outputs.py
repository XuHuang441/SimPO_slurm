from huggingface_hub import hf_hub_download

# 下载指定文件
file_path = hf_hub_download(
    repo_id="XuHuang/inpo_iter1",
    filename="all_outputs.json",
    cache_dir="/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter2/"
)

print("文件已下载到:", file_path)
