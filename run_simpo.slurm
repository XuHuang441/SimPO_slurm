#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:8
#SBATCH --job-name=mpo
#SBATCH --output=output.log
#SBATCH --mem=128G

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

ACCELERATE_LOG_LEVEL=info /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
--config_file accelerate_configs/deepspeed_zero3.yaml \
scripts/run_simpo.py \
training_configs/gemma-2-9b-it-simpo.yaml

notify "train complete" $?