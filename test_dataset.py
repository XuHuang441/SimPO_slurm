from datasets import load_from_disk

dataset = load_from_disk("/hai/scratch/fangwu97/xu/SimPO_slurm/data/sppo_iter1/pref/")
# dataset = load_from_disk("/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter2_skywork/all_outputs_bin_dataset")
print(dataset)
print(dataset.column_names)
print(dataset['train'][0])  # 查看第一条数据
