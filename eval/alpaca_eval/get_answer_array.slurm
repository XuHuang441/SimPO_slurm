#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=eval_array
#SBATCH --output=logs/output_eval_%A_%a.log
#SBATCH --mem=128G
#SBATCH --array=0-3

# run one gen and then upload

# sbatch eval/alpaca_eval/get_answer_array.slurm

# !!!! check sbatch array aligns with number of tasks
# 0-5 = 6

set -e

MODEL_NAMES=(
  "gemma-2-9b-it_mnpo_stage_2_armo_beta1_ratio0.85_eta0.005_weights0-1"
  "gemma-2-9b-it_mnpo_stage_2_armo_beta1_ratio0.85_eta0.005_weights0.25-0.75"
  "gemma-2-9b-it_mnpo_stage_2_armo_beta1_ratio0.85_eta0.005_weights0.75-0.25"
  "gemma-2-9b-it_mnpo_stage_2_armo_beta1_ratio0.85_eta0.005_weights1-0"
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
