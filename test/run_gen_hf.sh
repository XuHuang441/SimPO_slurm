source ~/anaconda3/etc/profile.d/conda.sh
conda activate /home/zbz5349/anaconda3/envs/inpo

conda run -n inpo python test_gen_hf.py \
--model_name_or_path "RLHFlow/LLaMA3-SFT" \
--dataset_name_or_path "RLHFlow/iterative-prompt-v1-iter1-20K" \
--output_dir "data" \
--K 8 --temperature 1.0 --local_index 0 \
--eos_ids 128009 