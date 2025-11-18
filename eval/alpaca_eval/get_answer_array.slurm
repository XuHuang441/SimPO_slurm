#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=eval_array
#SBATCH --output=output_eval_%A_%a.log
#SBATCH --mem=128G
#SBATCH --array=0-4

# run one gen and then upload
set -e

MODEL_NAMES=(
  "gemma-2-9b-it-mnpo-iter3-armo.yaml"
  "gemma-2-9b-it-mnpo-iter3-armo2.yaml"
  "gemma-2-9b-it-mnpo-iter2-armo-ratio0.86.yaml"
  "gemma-2-9b-it-mnpo-iter1-athene2.yaml"
  "gemma-2-9b-it-mnpo-iter1-athene3.yaml"
  "gemma-2-9b-it-mnpo-iter1-athene4.yaml"
)

# =============================
# 统一变量
# =============================
MODEL_NAME="gemma-2-9b-it_mnpo_stage_1_athene_beta10_ratio0.33_eta0.0075"
MODEL_PATH="/hai/scratch/fangwu97/xu/MNPO/outputs/$MODEL_NAME"
OUTPUT_JSON="res/${MODEL_NAME}.json"

# =============================
# 运行eval
# =============================
/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python /hai/scratch/fangwu97/xu/SimPO_slurm/eval/alpaca_eval/get_alpaca_answer_fast.py \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --conv_temp "gemma" \
    --cache_dir "/hai/scratch/fangwu97/xu/cache" \
    --tensor_parallel_size 2

# =============================
# 上传到 HuggingFace
# =============================
huggingface-cli upload XuHuang/inpo_iter1 "$OUTPUT_JSON"
