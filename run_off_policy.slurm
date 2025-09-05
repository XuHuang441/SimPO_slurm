#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:8
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
## precompute
#echo "iter2: start precompute"
#history_args=""
#if [ ${#history_paths[@]} -gt 0 ]; then
#    history_args="--history_paths ${history_paths[@]}"
#fi
#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=8 -m inpo_scripts.precompute_simpo_style \
#    --run_name "inpo_iter2" \
#    --train_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part2.jsonl" \
#    --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/inpo_iter2/pref" \
#    --ref_model google/gemma-2-9b-it \
#    --model_name_or_path "XuHuang/gemma-2-9b-it-inpo-iter1-bs128-lr5e-7" \
#    --loss_type inpo --lr_scheduler_type cosine \
#    $history_args \
#    --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache" \
#    --sanity_check False
#
#notify "iter2: precompute complete" $?

# train
echo "iter2: start training"

ACCELERATE_LOG_LEVEL=info /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    -m inpo_scripts.run_inpo \
    training_configs/gemma-2-9b-it-inpo-iter2.yaml

notify "iter2: train complete" $?

history_paths+=("/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it_off_policy_tdpo_stage_2_bs128_lr5e-7/")
#
echo "Completed iteration 2"


# #------------------------iter3------------------------
# precompute
echo "iter3: start precompute"
history_args=""
if [ ${#history_paths[@]} -gt 0 ]; then
    history_args="--history_paths ${history_paths[@]}"
fi
/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=8 -m inpo_scripts.precompute_simpo_style \
    --run_name "inpo_iter3" \
    --train_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part3.jsonl" \
    --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/inpo_iter3/pref" \
    --model_name_or_path "/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it_off_policy_tdpo_stage_2_bs128_lr5e-7/" \
    --ref_model google/gemma-2-9b-it \
    --loss_type inpo --lr_scheduler_type cosine \
    $history_args \
    --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache" \
    --sanity_check False

notify "iter3: precompute complete" $?

# train
echo "iter3: start training"
ACCELERATE_LOG_LEVEL=info  /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    -m inpo_scripts.run_inpo \
    training_configs/gemma-2-9b-it-inpo-iter3.yaml

notify "iter3: train complete" $?

history_paths+=("/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it_off_policy_tdpo_stage_3_bs128_lr5e-7/")

echo "Completed iteration 3"
