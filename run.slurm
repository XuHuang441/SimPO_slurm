#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:4
#SBATCH --job-name=mpo
#SBATCH --output=output.log
#SBATCH --mem=128G

set -e

source /hai/scratch/fangwu97/miniconda3/etc/profile.d/conda.sh
conda activate sim
export PYTHONPATH=$(pwd)

NTFY_TOPIC="https://ntfy.sh/xu-hai-update"

function notify() {
    STEP_NAME="$1"
    EXIT_CODE="$2"

    if [ "$EXIT_CODE" -eq 0 ]; then
        # 成功
        TITLE="✅ Success: $STEP_NAME"
        MESSAGE="Step '$STEP_NAME' completed successfully on job $SLURM_JOB_ID."
        PRIORITY="default"
        TAGS="white_check_mark"
    else
        # 失败
        TITLE="❌ Failure: $STEP_NAME"
        MESSAGE="Step '$STEP_NAME' failed with exit code $EXIT_CODE on job $SLURM_JOB_ID. Please check the logs."
        PRIORITY="high"
        TAGS="x"
    fi

    # 发送通知
    curl -s \
      -H "Title: $TITLE" \
      -H "Priority: $PRIORITY" \
      -H "Tags: $TAGS" \
      -d "$MESSAGE" \
      "$NTFY_TOPIC"

    # 如果脚本失败，则立即退出整个任务
    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "Exiting due to failure in step: $STEP_NAME"
        exit "$EXIT_CODE"
    fi
}

curl -s -H "Title: ▶️ Job Started" -d "Slurm Job $SLURM_JOB_ID ($SLURM_JOB_NAME) has started." "$NTFY_TOPIC"


history_paths=()

# divide dataset into 3 subsets with 20000 rows each.
#conda run -n sim python -m inpo_scripts.split_dataset

# ------------------------iter1------------------------
history_args=""

#  precompute # --config_file ./accelerate_configs/zero2.yaml
#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=4 -m inpo_scripts.precompute_simpo_style \
#     --run_name "inpo_iter1" \
#     --train_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1.jsonl" \
#     --output_dir "data/inpo_iter1/pref" \
#     --ref_model google/gemma-2-9b-it --last_model google/gemma-2-9b-it \
#     --loss_type inpo --lr_scheduler_type cosine \
#     $history_args \
#     --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache" \
#     --sanity_check False
#
#notify "iter1: precompute complete" $?
#
## train
#echo "iter1: start training"
#
#ACCELERATE_LOG_LEVEL=info /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
#    --config_file accelerate_configs/deepspeed_zero3.yaml \
#    -m inpo_scripts.run_inpo \
#    training_configs/gemma-2-9b-it-inpo-iter1.yaml \

#history_paths+=("/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it_inpo_stage_1/")
history_paths+=("XuHuang/gemma-2-9b-it-inpo-iter1-bs128-lr5e-7")

#echo "Completed iteration 1"

# ------------------------iter2------------------------
#echo "Starting iteration 2"
#
# on policy data gen
#echo "iter2: Starting on policy data gen"
#
#for SEED in 13 21 42 79 100
#  do
#     echo "Running decode with seed $SEED..."
#     stdbuf -oL -eL /hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -u -m on_policy_data_gen.decode \
#     --data_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part2.jsonl" \
#     --model "Timia123/simpo_inpo_iter1_aug13" \
#     --seed "$SEED" \
#     --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter2" \
#     --batch_size 96 \
#     --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache" \
#     --num_gpu 2 # Tensor Parallelism
#  done
#
#/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -m on_policy_data_gen.post_process \
#     --generation_file_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter2"

#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python -m on_policy_data_gen.reward_model_annotate \
#     --generation_file "/afs/.ir/users/f/a/fangwu97/.cache/huggingface/hub/models--XuHuang--inpo_iter1/snapshots/616f4bc181dead2cf5cf6bc4bf27e7f48bf067c8/all_outputs.json" \
#     --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter2" \
#     --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache"

#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python -m on_policy_data_gen.reward_model_annotate_skywork \
#     --generation_file "/afs/.ir/users/f/a/fangwu97/.cache/huggingface/hub/models--XuHuang--inpo_iter1/snapshots/616f4bc181dead2cf5cf6bc4bf27e7f48bf067c8/all_outputs.json" \
#     --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter2_skywork" \
#     --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache"
#
#notify "iter2: reward_model_annotate_skywork complete" $?

# precompute
echo "iter2: start precompute"
history_args=""
if [ ${#history_paths[@]} -gt 0 ]; then
    history_args="--history_paths ${history_paths[@]}"
fi
/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=4 -m inpo_scripts.precompute_simpo_style \
    --run_name "inpo_iter2" \
    --train_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter2_skywork/all_outputs_bin_dataset" \
    --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/inpo_iter2/pref_skywork" \
    --ref_model google/gemma-2-9b-it \
    --loss_type inpo --lr_scheduler_type cosine \
    $history_args \
    --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache" \
    --sanity_check True

#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=4 -m inpo_scripts.precompute_simpo_style \
#    --run_name "inpo_iter2" \
#    --train_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part2_test.jsonl" \
#    --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/inpo_iter2/pref_rm/test" \
#    --ref_model google/gemma-2-9b-it \
#    --loss_type inpo --lr_scheduler_type cosine \
#    $history_args \
#    --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache" \
#    --sanity_check False

# train
#echo "iter2: start training"
#
#ACCELERATE_LOG_LEVEL=info /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
#    --config_file accelerate_configs/deepspeed_zero3.yaml \
#    -m inpo_scripts.run_inpo \
#    training_configs/gemma-2-9b-it-inpo-iter2.yaml
#
#notify "iter2: train complete" $?
#
##
#history_paths+=("/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it_inpo_stage_2_bs_128_lr_5e-7_eval15/")
#
#echo "Completed iteration 2"

# #------------------------iter3------------------------
# on policy data gen
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
#     --batch_size 96 \
#     --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache" \
#     --num_gpu 8 # Tensor Parallelism
#  done
#
#/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -m on_policy_data_gen.post_process \
#     --generation_file_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter3"
# dont forget to check empty strings
#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python -m on_policy_data_gen.reward_model_annotate \
#     --generation_file "/afs/.ir/users/f/a/fangwu97/.cache/huggingface/hub/models--XuHuang--inpo_iter1/snapshots/fc8024b5e4353e5f74954d4133d20813267e77b8/all_outputs.json" \
#     --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter3"
#
## precompute
#echo "iter3: start precompute"
#history_args=""
#if [ ${#history_paths[@]} -gt 0 ]; then
#    history_args="--history_paths ${history_paths[@]}"
#fi
#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=4 -m inpo_scripts.precompute_simpo_style \
#    --run_name "inpo_iter3" \
#    --train_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/inpo_iter3" \
#    --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/inpo_iter3/pref" \
#    --ref_model google/gemma-2-9b-it \
#    --loss_type inpo --lr_scheduler_type cosine \
#    $history_args \
#    --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache" \
#    --sanity_check False

# train
#echo "iter3: start training"
#ACCELERATE_LOG_LEVEL=info  /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
#    --config_file accelerate_configs/deepspeed_zero3.yaml \
#    -m inpo_scripts.run_inpo \
#    training_configs/gemma-2-9b-it-inpo-iter3.yaml
#notify "iter3: training complete" $?
#history_paths+=("/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it_inpo_stage_3/")
#
#echo "Completed iteration 3"
