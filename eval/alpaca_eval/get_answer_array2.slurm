#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=eval_array2
#SBATCH --output=logs/output_eval_%A_%a.log
#SBATCH --mem=128G
#SBATCH --array=0-2

# run one gen and then upload

# customize the pid
# sbatch --dependency=afterok:40127_1 eval/alpaca_eval/get_answer_array2.slurm

# !!!! check sbatch array aligns with number of tasks
# 0-5 = 6

set -e

MODEL_NAMES=(
  "gemma-2-9b-it_mnpo_stage_2_skywork_beta5_ratio0.8_eta0.01_weights0.25-0.75_3pl"
  "gemma-2-9b-it_mnpo_stage_2_skywork_beta10_ratio0.8_eta0.01_weights0.2-0.8_3pl"
  "gemma-2-9b-it_mnpo_stage_2_skywork_beta10_ratio0.75_eta0.01_weights0.25-0.75_3pl"
)

MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}
MODEL_PATH="/hai/scratch/fangwu97/xu/MNPO/outputs/$MODEL_NAME"
OUTPUT_JSON="res/${MODEL_NAME}.json"

echo "evaluating model: $MODEL_NAME"
echo "model path: $MODEL_PATH"

/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python /hai/scratch/fangwu97/xu/SimPO_slurm/eval/alpaca_eval/get_alpaca_answer_fast.py \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --conv_temp "gemma" \
    --cache_dir "/hai/scratch/fangwu97/xu/cache" \
    --tensor_parallel_size 2

huggingface-cli upload XuHuang/inpo_iter1 "$OUTPUT_JSON"
