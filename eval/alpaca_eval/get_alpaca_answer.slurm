#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=mpo_eval
#SBATCH --output=output_eval.log
#SBATCH --mem=128G

model_name="gemma-2-9b-it_inpo_stage_1_bs_128_lr_5e-7_sep10_ckpt155"
model_path="/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it_inpo_stage_1_bs_128_lr_5e-7_eval15_sep10/checkpoint-155"
# CUDA_VISIBLE_DEVICES=3 python get_alpaca_answer.py --model_name $model_name --model_path $model_path --conv_temp "gemma"

/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python /hai/scratch/fangwu97/xu/SimPO_slurm/eval/alpaca_eval/get_alpaca_answer_fast.py \
    --model_name $model_name \
    --model_path $model_path \
    --conv_temp "gemma" \
    --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/outputs" \
    --tensor_parallel_size 4



