#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=eval
#SBATCH --output=output_eval.log
#SBATCH --mem=128G

#  request 2 gpus and run 3 tasks sequentially

/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python /hai/scratch/fangwu97/xu/SimPO_slurm/eval/alpaca_eval/get_alpaca_answer_fast.py \
    --model_name gemma-2-9b-it_mnpo_stage_2_armo_ratio0.83_eta0.005 \
    --model_path /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_armo_ratio0.83_eta0.005 \
    --conv_temp "gemma" \
    --cache_dir "/hai/scratch/fangwu97/xu/cache" \
    --tensor_parallel_size 2

/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python /hai/scratch/fangwu97/xu/SimPO_slurm/eval/alpaca_eval/get_alpaca_answer_fast.py \
    --model_name gemma-2-9b-it_mnpo_stage_2_armo_ratio0.84_eta0.005 \
    --model_path /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_armo_ratio0.84_eta0.005 \
    --conv_temp "gemma" \
    --cache_dir "/hai/scratch/fangwu97/xu/cache" \
    --tensor_parallel_size 2

/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python /hai/scratch/fangwu97/xu/SimPO_slurm/eval/alpaca_eval/get_alpaca_answer_fast.py \
    --model_name gemma-2-9b-it_mnpo_stage_2_armo_ratio0.86_eta0.005 \
    --model_path /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_armo_ratio0.86_eta0.005 \
    --conv_temp "gemma" \
    --cache_dir "/hai/scratch/fangwu97/xu/cache" \
    --tensor_parallel_size 2