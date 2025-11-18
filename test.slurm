#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:1
#SBATCH --job-name=test
#SBATCH --output=output_test.log
#SBATCH --mem=128G

/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python test_local_model.py