from datasets import load_from_disk

# history0=ref at iter1 
dataset = load_from_disk("/home/zbz5349/zhiwei/Multi-player-Nash-Policy-Optimization-main/SimPO/data/inpo_iter2/pref")

# dataset = load_from_disk("/home/zbz5349/zhiwei/Multi-player-Nash-Policy-Optimization-main/SimPO/datasets/gemma2_ultrafeedback")

print(dataset["train"][0])
# 查看所有字段名（即 key）
print(dataset.features)

print(dataset[2])  # 打印第一条数据的内容
