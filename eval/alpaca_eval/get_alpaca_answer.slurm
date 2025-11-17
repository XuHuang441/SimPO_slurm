#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=eval
#SBATCH --output=output_eval.log
#SBATCH --mem=128G

srun --exclusive --gpus-per-task=2 \
/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python /hai/scratch/fangwu97/xu/SimPO_slurm/eval/alpaca_eval/get_alpaca_answer_fast.py \
    --model_name gemma-2-9b-it_mnpo_stage_2_skywork_ratio0.8_eta0.005 \
    --model_path /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_skywork_ratio0.8_eta0.005 \
    --conv_temp "gemma" \
    --cache_dir "/hai/scratch/fangwu97/xu/cache" \
    --tensor_parallel_size 2 &

#srun --exclusive --gpus-per-task=2 \
#/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python /hai/scratch/fangwu97/xu/SimPO_slurm/eval/alpaca_eval/get_alpaca_answer_fast.py \
#    --model_name gemma-2-9b-it_mnpo_stage_2_armo_puredpoloss_beta5 \
#    --model_path /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_armo_puredpoloss_beta5 \
#    --conv_temp "gemma" \
#    --cache_dir "/hai/scratch/fangwu97/xu/cache" \
#    --tensor_parallel_size 2 &

wait