#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:1
#SBATCH --job-name=upload_model
#SBATCH --output=output_upload.log

/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python upload_hf_model.py