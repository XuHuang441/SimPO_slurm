#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=eval_new
#SBATCH --output=output_eval_new.log
#SBATCH --mem=128G

# run one gen and then upload
set -e

# =============================
# 统一变量
# =============================
MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
OUTPUT_JSON="res/${MODEL_NAME}.json"

# =============================
# 运行eval
# =============================
/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python /hai/scratch/fangwu97/xu/SimPO_slurm/eval/alpaca_eval/get_alpaca_answer_new.py \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_NAME" \
    --cache_dir "/hai/scratch/fangwu97/xu/cache" \
    --tensor_parallel_size 2

# =============================
# 上传到 HuggingFace
# =============================
huggingface-cli upload XuHuang/inpo_iter1 "$OUTPUT_JSON"
