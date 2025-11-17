#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=athene
#SBATCH --output=output_athene.log

export HF_HOME=/hai/scratch/fangwu97/hf_cache

/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python -u -m on_policy_data_gen.rm_athene \
    --input_file /hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1_20k/gemma2_ufb_part1_split1.jsonl \
    --output_file /hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter1_athene_scored.jsonl \
    --cache_dir /hai/scratch/fangwu97/xu/cache \
    --batch_size 8
