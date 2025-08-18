import json
from datasets import load_dataset
import argparse
import os
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--generation_file_dir", type=str, help="Diretory containing the generation files", default="datasets/gemma2_ultrafeedback")
parser.add_argument('--model', type=str, default="google/gemma-2-9b-it",
                    help='Path to the LLM model')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument('--num_gpu', type=int, default=4)
parser.add_argument('--cache_dir', type=str, default=None,
                    help='Cache directory for model and dataset')
parser.add_argument('--temperature', type=float, default=0.8,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p probability for sampling')
parser.add_argument('--max_tokens', type=int, default=4096,
                    help='Maximum number of tokens to generate')

args = parser.parse_args()

print(args)

with open(args.generation_file_dir, "r", encoding="utf-8") as f:
    data = json.load(f)

empty_strs = []

for i, sample in enumerate(data):
    responses = sample.get("all_generated_responses", [])
    for j, resp in enumerate(responses):
        if isinstance(resp, str) and resp.strip() == "":
            empty_strs.append((i, j))  # i = 第几个样本, j = 第几个response

seeds = [13, 21, 42, 79, 100]

if empty_strs:
    print(f"Found {len(empty_strs)} empty strings. Starting regeneration...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.num_gpu,
        download_dir=args.cache_dir,
        gpu_memory_utilization=0.9,  # 允许 VLLM 使用 90% 的 GPU 显存
    )
    tokenizer = llm.get_tokenizer()
    for i, j in empty_strs:
        print(f"Sample {i}'s all_generated_responses[{j}] is empty. Regenerating with seed {seeds[j]}...")

        sampling_params = SamplingParams(temperature=args.temperature,
                                         top_p=args.top_p,
                                         max_tokens=args.max_tokens,
                                         seed=seeds[j], )

        # 2. Get the original prompt for this sample
        prompt_text = data[i].get('prompt')
        if not prompt_text:
            print(f"  - WARNING: Skipping sample {i} because its 'prompt' is missing.")
            continue

        # Apply the chat template to format the prompt correctly
        messages = [
            {"role": "user", "content": prompt_text}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 3. Regenerate the response
        outputs = llm.generate(formatted_prompt, sampling_params, use_tqdm=False)

        # 4. Extract and fill in the newly generated response
        if outputs:
            generated_text = outputs[0].outputs[0].text
            print(f"  - Generated text: {generated_text}")
            data[i]["all_generated_responses"][j] = generated_text
            print(f"  - Successfully regenerated and filled.")
        else:
            print(f"  - WARNING: Regeneration failed for sample {i}, response {j}.")

        # --- Save the corrected data back to the file ---
        print("Regeneration complete. Saving the updated data...")
        with open(args.generation_file_dir, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print("File saved.")

else:
    print("No empty strings found.")
