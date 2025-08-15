#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=mpo
#SBATCH --output=output.log

source /hai/scratch/fangwu97/miniconda3/etc/profile.d/conda.sh
conda activate sim
export PYTHONPATH=$(pwd)

history_paths=()

# divide dataset into 3 subsets with 20000 rows each.
#conda run -n sim python -m inpo_scripts.split_dataset

# ------------------------iter1------------------------
history_args=""

#  precompute # --config_file ./accelerate_configs/zero2.yaml
/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=2 -m inpo_scripts.precompute_simpo_style \
     --run_name "inpo_iter1" \
     --train_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1.jsonl" \
     --output_dir "data/inpo_iter1/pref" \
     --ref_model google/gemma-2-9b-it --last_model google/gemma-2-9b-it \
     --loss_type inpo --lr_scheduler_type cosine \
     $history_args \
     --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache" \
     --sanity_check True

# train
#echo "iter1: start training"
#
#ACCELERATE_LOG_LEVEL=info /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
#    --config_file accelerate_configs/deepspeed_zero3.yaml \
#    -m inpo_scripts.run_inpo \
#    training_configs/gemma-2-9b-it-inpo-iter1.yaml \

#history_paths+=("/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it_inpo_stage_1/")

#echo "Completed iteration 1"

# ------------------------iter2------------------------
#echo "Starting iteration 2"
#
## on policy data gen
#echo "iter2: Starting on policy data gen"
#
#for SEED in 13 21 42 79 100
#  do
#     echo "Running decode with seed $SEED..."
#     stdbuf -oL -eL /hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -u -m on_policy_data_gen.decode \
#     --data_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part2.jsonl" \
#     --model "/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it_inpo_stage_1" \
#     --seed "$SEED" \
#     --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter2" \
#     --batch_size 8192 \
#     --num_gpu 8 # Tensor Parallelism
#  done
#
#/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -m on_policy_data_gen.post_process \
#     --generation_file_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter2"
#
#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python -m on_policy_data_gen.reward_model_annotate \
#     --generation_file "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter2/all_outputs.json" \
#     --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter2"
#
## precompute
#echo "iter2: start precompute"
#history_args=""
#if [ ${#history_paths[@]} -gt 0 ]; then
#    history_args="--history_paths ${history_paths[@]}"
#fi
#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=8 -m inpo_scripts.precompute \
#    --run_name "inpo_iter2" \
#    --train_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter2" \
#    --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/inpo_iter2/pref" \
#    --ref_model google/gemma-2-9b-it \
#    --loss_type inpo --lr_scheduler_type cosine \
#    $history_args \
#    --sanity_check False
#
## train
#echo "iter2: start training"
#
#ACCELERATE_LOG_LEVEL=info /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
#    --config_file accelerate_configs/deepspeed_zero3.yaml \
#    -m inpo_scripts.run_inpo \
#    training_configs/gemma-2-9b-it-inpo-iter2.yaml \
#
#history_paths+=("/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it_inpo_stage_2/")
#
#echo "Completed iteration 2"
#
# #------------------------iter3------------------------
## on policy data gen
#echo "iter3: Starting on policy data gen"
#
#for SEED in 13 21 42 79 100
#  do
#     echo "Running decode with seed $SEED..."
#     stdbuf -oL -eL /hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -u -m on_policy_data_gen.decode \
#     --data_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part3.jsonl" \
#     --model "/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it_inpo_stage_2/" \
#     --seed "$SEED" \
#     --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter3" \
#     --batch_size 8192 \
#     --num_gpu 8 # Tensor Parallelism
#  done
#
#/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -m on_policy_data_gen.post_process \
#     --generation_file_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter3"
#
#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python -m on_policy_data_gen.reward_model_annotate \
#     --generation_file "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter3/all_outputs.json" \
#     --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter3"
#
## precompute
#echo "iter3: start precompute"
#history_args=""
#if [ ${#history_paths[@]} -gt 0 ]; then
#    history_args="--history_paths ${history_paths[@]}"
#fi
#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=8 -m inpo_scripts.precompute \
#    --run_name "inpo_iter3" \
#    --train_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter3" \
#    --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/inpo_iter3/pref" \
#    --ref_model google/gemma-2-9b-it \
#    --loss_type inpo --lr_scheduler_type cosine \
#    $history_args \
#    --sanity_check False
#
## train
#echo "iter3: start training"
#ACCELERATE_LOG_LEVEL=info  /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
#    --config_file accelerate_configs/deepspeed_zero3.yaml \
#    -m inpo_scripts.run_inpo \
#    training_configs/gemma-2-9b-it-inpo-iter3.yaml \
#
#history_paths+=("/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it_inpo_stage_3/")
#
#echo "Completed iteration 3"
