from pprint import pprint

from datasets import load_dataset

from datasets import load_from_disk

dataset = load_from_disk("/hai/scratch/fangwu97/xu/SimPO_slurm/data/inpo_iter2/pref_skywork")

print(dataset.features)

# ref_wrong = 0
# ref_total = 0
# hist0_wrong = 0
# hist0_total = 0
#
# for example in dataset:
#     ref_c = example["reference_chosen_logps"]
#     ref_r = example["reference_rejected_logps"]
#     hist0_c = example["history0_chosen_logps"]
#     hist0_r = example["history0_rejected_logps"]
#
#     if ref_c is not None and ref_r is not None:
#         ref_total += 1
#         if ref_c < ref_r:  # reference 觉得 chosen 更差
#             ref_wrong += 1
#
#     if hist0_c is not None and hist0_r is not None:
#         hist0_total += 1
#         if hist0_c < hist0_r:  # history0 觉得 chosen 更差
#             hist0_wrong += 1
#
# print(f"Reference model 反直觉样本: {ref_wrong}/{ref_total} ({ref_wrong/ref_total:.2%})")
# print(f"History0 model 反直觉样本: {hist0_wrong}/{hist0_total} ({hist0_wrong/hist0_total:.2%})")


