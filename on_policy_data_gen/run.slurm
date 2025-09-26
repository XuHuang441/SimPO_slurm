#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:1
#SBATCH --job-name=test
#SBATCH --output=output_test.log

export HF_HOME=/hai/scratch/fangwu97/hf_cache

/hai/scratch/fangwu97/miniconda3/envs/llmblender/bin/python preference_model_annotate.py \
--input_file hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter2/all_outputs.json \
--output_file hai/scratch/fangwu97/xu/SimPO_slurm/on_policy_data_gen/preference_data.json