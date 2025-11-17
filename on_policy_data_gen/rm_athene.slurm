#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=athene
#SBATCH --output=output_athene.log

export HF_HOME=/hai/scratch/fangwu97/hf_cache

/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python -u -m on_policy_data_gen.rm_athene