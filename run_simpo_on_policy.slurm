#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p hai-lo
#SBATCH --gres=gpu:4
#SBATCH --job-name=mpo
#SBATCH --output=output.log
#SBATCH --mem=128G

set -e

source /hai/scratch/fangwu97/miniconda3/etc/profile.d/conda.sh
conda activate sim
export PYTHONPATH=$(pwd)

# divide dataset into 3 subsets with 20000 rows each.
conda run -n inpo python -m inpo_scripts.split_dataset

# ------------------------iter1------------------------
echo "iter1: start training"

ACCELERATE_LOG_LEVEL=info /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
--config_file accelerate_configs/deepspeed_zero3.yaml \
scripts/run_simpo.py \
training_configs/gemma-2-9b-it-simpo_iter1_on_policy.yaml

# ------------------------iter2------------------------
echo "Starting iteration 2"

# on policy data gen, make sure to check if there's empty outputs
echo "iter2: Starting on policy data gen"

for SEED in 13 21 42 79 100
  do
     echo "Running decode with seed $SEED..."
     stdbuf -oL -eL /hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -u -m on_policy_data_gen.decode \
     --data_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part2.jsonl" \
     --model "/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it-simpo-iter1_on_policy" \
     --seed "$SEED" \
     --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/simpo_iter2" \
     --batch_size 96 \
     --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache" \
     --num_gpu 4 # Tensor Parallelism
  done

/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -m on_policy_data_gen.post_process \
     --generation_file_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/simpo_iter2"

#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python -m on_policy_data_gen.reward_model_annotate \
#     --generation_file "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/simpo_iter2/all_outputs.json" \
#     --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/simpo_iter2"
#
#echo "iter2: start training"
#
#ACCELERATE_LOG_LEVEL=info /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
#--config_file accelerate_configs/deepspeed_zero3.yaml \
#scripts/run_simpo.py \
#training_configs/gemma-2-9b-it-simpo_iter2_on_policy.yaml
#
#echo "Completed iteration 2"
#
## ------------------------iter3------------------------
#echo "Starting iteration 3"
#
## on policy data gen, make sure to check if there's empty outputs
#echo "iter3: Starting on policy data gen"
#
#for SEED in 13 21 42 79 100
#  do
#     echo "Running decode with seed $SEED..."
#     stdbuf -oL -eL /hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -u -m on_policy_data_gen.decode \
#     --data_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part3.jsonl" \
#     --model "/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it-simpo-iter2_on_policy" \
#     --seed "$SEED" \
#     --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/simpo_iter3" \
#     --batch_size 96 \
#     --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache" \
#     --num_gpu 8 # Tensor Parallelism
#  done
#
#/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -m on_policy_data_gen.post_process \
#     --generation_file_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/simpo_iter3"
#
#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python -m on_policy_data_gen.reward_model_annotate \
#     --generation_file "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/simpo_iter3/all_outputs.json" \
#     --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/simpo_iter3"
#
#echo "iter3: start training"
#
#ACCELERATE_LOG_LEVEL=info /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
#--config_file accelerate_configs/deepspeed_zero3.yaml \
#scripts/run_simpo.py \
#training_configs/gemma-2-9b-it-simpo_iter3_on_policy.yaml
#
#echo "Completed iteration 3"