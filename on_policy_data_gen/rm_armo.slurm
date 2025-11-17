#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:1
#SBATCH --job-name=armo
#SBATCH --output=output_armo.log

# somehow the armo rm can not run on multigpu

export HF_HOME=/hai/scratch/fangwu97/hf_cache

/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python -u -m on_policy_data_gen.rm_armo_new