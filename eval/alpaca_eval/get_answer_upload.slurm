#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=eval
#SBATCH --output=output_eval.log
#SBATCH --mem=128G

# run one gen and then upload
set -e

# =============================
# 统一变量
# =============================
MODEL_NAME="Llama-3.1-Tulu-3-70B-DPO"
MODEL_PATH="/hai/scratch/fangwu97/huggingface/hub/models--allenai--Llama-3.1-Tulu-3-70B-DPO/snapshots/2dab5183da9252bedf6039a0b09c86b9ca4d0ba9/"
OUTPUT_JSON="res/${MODEL_NAME}.json"

# =============================
# 运行eval
# =============================
/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python /hai/scratch/fangwu97/xu/SimPO_slurm/eval/alpaca_eval/get_alpaca_answer_fast.py \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --conv_temp "tulu" \
    --cache_dir "/hai/scratch/fangwu97/xu/cache" \
    --tensor_parallel_size 2

# =============================
# 上传到 HuggingFace
# =============================
huggingface-cli upload XuHuang/inpo_iter1 "$OUTPUT_JSON"
