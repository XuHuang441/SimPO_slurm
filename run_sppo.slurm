#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:4
#SBATCH --job-name=mpo
#SBATCH --output=output_sppo.log
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

#  precompute # --config_file ./accelerate_configs/zero2.yaml
#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=4 -m inpo_scripts.precompute_sppo \
#     --run_name "inpo_iter1" \
#     --train_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1.jsonl" \
#     --output_dir "data/sppo_iter1/pref/train2" \
#     --model_name_or_path google/gemma-2-9b-it \
#     --ref_model google/gemma-2-9b-it --last_model google/gemma-2-9b-it \
#     --loss_type inpo --lr_scheduler_type cosine \
#     --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache" \
#     --sanity_check False
#
#notify "iter1: precompute complete" $?

#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=4 -m inpo_scripts.precompute_sppo \
#   --run_name "inpo_iter1" \
#   --train_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1_test.jsonl" \
#   --output_dir "data/sppo_iter1/pref/test" \
#   --ref_model google/gemma-2-9b-it \
#   --model_name_or_path google/gemma-2-9b-it \
#   --loss_type inpo --lr_scheduler_type cosine \
#   $history_args \
#   --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache" \
#   --sanity_check False

# train
echo "iter1: start training"

ACCELERATE_LOG_LEVEL=info /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    -m inpo_scripts.run_sppo \
    training_configs/gemma-2-9b-it-sppo-iter1.yaml

notify "iter1: train complete" $?