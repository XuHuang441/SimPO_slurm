from huggingface_hub import hf_hub_download

# 下载指定文件
file_path = hf_hub_download(
    repo_id="XuHuang/inpo_iter1",
    filename="all_outputs.json",
)

print("文件已下载到:", file_path)
