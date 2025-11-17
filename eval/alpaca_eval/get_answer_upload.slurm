#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=eval
#SBATCH --output=output_eval.log
#SBATCH --mem=128G

set -e

# =============================
# 统一变量
# =============================
MODEL_NAME="gemma-2-9b-it_mnpo_stage_3_armo_ratio0.85_eta0.005_weights0-1"
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
