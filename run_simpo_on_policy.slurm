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

# divide dataset into 3 subsets with 20000 rows each.
#conda run -n sim python -m inpo_scripts.split_dataset

# ------------------------iter1------------------------
#echo "iter1: start training"
#
#ACCELERATE_LOG_LEVEL=info /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
#--config_file accelerate_configs/deepspeed_zero3.yaml \
#scripts/run_simpo.py \
#training_configs/gemma-2-9b-it-simpo_iter1_on_policy.yaml
#
## ------------------------iter2------------------------
#echo "Starting iteration 2"
#
## on policy data gen, make sure to check if there's empty outputs
#echo "iter2: Starting on policy data gen"
#
#for SEED in 13 21 42 79 100
#  do
#     echo "Running decode with seed $SEED..."
#     stdbuf -oL -eL /hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -u -m on_policy_data_gen.decode \
#     --data_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part2.jsonl" \
#     --model "/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it-simpo-iter1_on_policy" \
#     --seed "$SEED" \
#     --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/simpo_iter2" \
#     --batch_size 96 \
#     --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache" \
#     --num_gpu 4 # Tensor Parallelism
#  done
#
#/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -m on_policy_data_gen.post_process \
#     --generation_file_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/simpo_iter2"

/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python -m on_policy_data_gen.reward_model_annotate \
     --generation_file "/afs/.ir/users/f/a/fangwu97/.cache/huggingface/hub/models--XuHuang--inpo_iter1/snapshots/4dbda9026c77c26c8d6d83873f79ab112f4e7e78/all_outputs.json" \
     --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/simpo_iter2"
notify "iter2: reward_model_annotate" $?
echo "iter2: start training"

ACCELERATE_LOG_LEVEL=info /hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch \
--config_file accelerate_configs/deepspeed_zero3.yaml \
scripts/run_simpo.py \
training_configs/gemma-2-9b-it-simpo_iter2_on_policy.yaml
notify "iter2: Training" $?
echo "Completed iteration 2"

# ------------------------iter3------------------------
echo "Starting iteration 3"

# on policy data gen, make sure to check if there's empty outputs
echo "iter3: Starting on policy data gen"

for SEED in 13 21 42 79 100
  do
     echo "Running decode with seed $SEED..."
     stdbuf -oL -eL /hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -u -m on_policy_data_gen.decode \
     --data_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part3.jsonl" \
     --model "/hai/scratch/fangwu97/xu/SimPO_slurm/outputs/gemma-2-9b-it-simpo-iter2_on_policy" \
     --seed "$SEED" \
     --output_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/simpo_iter3" \
     --batch_size 96 \
     --cache_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/cache" \
     --num_gpu 8 # Tensor Parallelism
  done

notify "iter3: All decode seeds completed" $?

/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -m on_policy_data_gen.post_process \
     --generation_file_dir "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/simpo_iter3"
notify "iter3: Post-processing" $?


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
curl -s -H "Title: 🎉 Job Finished" -d "Slurm Job $SLURM_JOB_ID ($SLURM_JOB_NAME) has finished successfully." "$NTFY_TOPIC"
