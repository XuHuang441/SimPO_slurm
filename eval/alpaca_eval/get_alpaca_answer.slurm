#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=mpo_eval
#SBATCH --output=output.log
#SBATCH --mem=128G

# change model_name and model_path to your own model and you are ready to go!
source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate /hai/scratch/fangwu97/miniconda3/envs/inpo

model_name="gemma-2-9b-it_off_policy_tdpo_stage_2_bs128_lr5e-7"
model_path="/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it_off_policy_tdpo_stage_2_bs128_lr5e-7/"
# CUDA_VISIBLE_DEVICES=3 python get_alpaca_answer.py --model_name $model_name --model_path $model_path --conv_temp "gemma"

python get_alpaca_answer_fast.py \
    --model_name $model_name \
    --model_path $model_path \
    --conv_temp "gemma" \
    --tensor_parallel_size 2