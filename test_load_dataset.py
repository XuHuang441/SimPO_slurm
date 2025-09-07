
from datasets import DatasetDict, load_from_disk

dataset_path="/hai/scratch/fangwu97/xu/SimPO_slurm/data/inpo_iter2/pref_rm"

raw_datasets = load_from_disk(dataset_path)

if not isinstance(raw_datasets, DatasetDict):
    raw_datasets = DatasetDict({"train": raw_datasets})

train_dataset = None
eval_dataset = None
for split_name in raw_datasets.keys():
    if "train" in split_name:
        train_dataset = raw_datasets[split_name]
    elif "test" in split_name or "eval" in split_name:
        eval_dataset = raw_datasets[split_name]

if train_dataset is None:
    raise ValueError(
        f"No training split found in the dataset. Available splits: {list(raw_datasets.keys())}"
    )

print(f"Using '{next(k for k, v in raw_datasets.items() if v is train_dataset)}' for training.")
if eval_dataset:
    print(f"Using '{next(k for k, v in raw_datasets.items() if v is eval_dataset)}' for evaluation.")
else:
    print("No evaluation split found or selected.")

print(f"Loaded dataset splits: {list(raw_datasets.keys())}")