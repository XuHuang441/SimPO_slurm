# /home/zbz5349/anaconda3/envs/sim/bin/pip

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /home/zbz5349/anaconda3/envs/sim
export PYTHONPATH=$(pwd)

history_paths=()

# ------------------------iter1------------------------
history_args=""
if [ ${#history_paths[@]} -gt 0 ]; then
    history_args="--history_paths ${history_paths[@]}"
fi
# # precompute # --config_file ./accelerate_configs/zero2.yaml
# conda run -n sim accelerate launch --num_processes=4 -m inpo_scripts.precompute \
#     --run_name "inpo_iter1" \
#     --train_dir "/home/zbz5349/zhiwei/Multi-player-Nash-Policy-Optimization-main/SimPO/data/gemma2_ufb_part1.jsonl" \
#     --output_dir "/home/zbz5349/zhiwei/Multi-player-Nash-Policy-Optimization-main/SimPO/data/inpo_iter1/pref" \
#     --ref_model google/gemma-2-9b-it --last_model google/gemma-2-9b-it \
#     --loss_type inpo --lr_scheduler_type cosine \
#     $history_args \
#     --sanity_check True

# train
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1,2,3 ACCELERATE_LOG_LEVEL=info conda run -n sim accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    -m inpo_scripts.run_inpo \
    /home/zbz5349/zhiwei/Multi-player-Nash-Policy-Optimization-main/SimPO/training_configs/gemma-2-9b-it-inpo-iter1.yaml \

# ------------------------iter2------------------------
# on policy data gen
# for SEED in 13 21 42 79 100
# do
#     echo "Running decode with seed $SEED..."
#     CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n inpo python -m on_policy_data_gen.decode \
#     --data_dir "/home/zbz5349/zhiwei/Multi-player-Nash-Policy-Optimization-main/SimPO/data/gemma2_ufb_part2.jsonl" \
#     --seed "$SEED" \
#     --sanity_check True \
#     --output_dir "datasets/gemma2_ultrafeedback/inpo_iter2" \
#     --num_gpu 4 # Tensor Parallelism

# done

# conda run -n inpo python -m on_policy_data_gen.post_process \
#     --generation_file_dir "datasets/gemma2_ultrafeedback/inpo_iter2"

# conda run -n sim python -m on_policy_data_gen.reward_model_annotate \
#     --generation_file "datasets/gemma2_ultrafeedback/inpo_iter2/all_outputs.json" \
#     --output_dir "datasets/gemma2_ultrafeedback/inpo_iter2"

# precompute
# conda run -n sim accelerate launch --num_processes=4 -m inpo_scripts.precompute \
#     --run_name "inpo_iter2" \
#     --train_dir "datasets/gemma2_ultrafeedback/inpo_iter2" \
#     --output_dir "/home/zbz5349/zhiwei/Multi-player-Nash-Policy-Optimization-main/SimPO/data/inpo_iter2/pref" \
#     --ref_model google/gemma-2-9b-it --last_model google/gemma-2-9b-it \
#     --loss_type inpo --lr_scheduler_type cosine \
#     $history_args \
#     --sanity_check True

# train
# ACCELERATE_LOG_LEVEL=info conda run -n sim accelerate launch \
# --config_file accelerate_configs/deepspeed_zero3.yaml \
# scripts/run_inpo.py \
# training_configs/gemma-2-9b-it-inpo.yaml \
# --set_values model_name_or_path=./outputs/gemma-2-9b-it_inpo_stage_1 \
#              dataset_name=./data/gemma2_ufb_part2.jsonl \
#              output_dir=./outputs/gemma-2-9b-it_inpo_stage_2 \
#              run_name=gemma-2-9b-it_inpo_stage_2 \
#              learning_rate=4.0e-7

# ------------------------iter3------------------------
# on policy data gen

# precompute
# conda run -n sim python precompute.py \
#     --reference_model_path "google/gemma-2-9b-it" \
#     --history_model_paths "./outputs/inpo_stage_1" "./outputs/inpo_stage_2" \
#     --input_dataset_name "princeton-nlp/gemma2-ultrafeedback-armorm" \
#     --output_dataset_path "./data/final_precomputed_dataset" \
#     --per_device_batch_size 4 \
#     --torch_dtype "bfloat16"

# train
# ACCELERATE_LOG_LEVEL=info conda run -n sim accelerate launch \
# --config_file accelerate_configs/deepspeed_zero3.yaml \
# scripts/run_inpo.py \
# training_configs/gemma-2-9b-it-inpo.yaml \
# --set_values model_name_or_path=./outputs/gemma-2-9b-it_inpo_stage_2 \
#              dataset_name=./data/gemma2_ufb_part3.jsonl \
#              output_dir=./outputs/gemma-2-9b-it_inpo_stage_3 \
#              run_name=gemma-2-9b-it_inpo_stage_3 \
#              learning_rate=2.0e-7