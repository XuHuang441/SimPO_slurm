from datasets import load_from_disk

dataset = load_from_disk("/hai/scratch/fangwu97/xu/SimPO_slurm/data/inpo_iter2/pref/")
print(dataset)
print(dataset.column_names)
print(dataset['train'][0])  # 查看第一条数据
