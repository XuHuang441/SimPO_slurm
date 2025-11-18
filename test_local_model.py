# check_model_nan.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

MODEL_PATH = "/hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_armo_ratio0.85_eta0.005"

def check_params_nan_inf(model):
    has_bad = False
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if torch.isnan(param).any():
            print(f"[NaN] in param: {name}")
            has_bad = True
            break
        if torch.isinf(param).any():
            print(f"[Inf] in param: {name}")
            has_bad = True
            break
    if not has_bad:
        print("✅ No NaN/Inf found in model parameters.")
    return has_bad

def check_forward_nan_inf(model, tokenizer):
    model.eval()
    text = "Hello, this is a test."
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits
    print("logits shape:", logits.shape)
    print("logits has NaN:", torch.isnan(logits).any().item())
    print("logits has Inf:", torch.isinf(logits).any().item())

def main():
    model_path = Path(MODEL_PATH)
    print(f"Loading model from: {model_path}")

    # 如果你正常训练用的是 bfloat16 / fp16，这里也可以指定 dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,   # 或者 torch.float16 / torch.float32，看你当时怎么训的
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    print("=== Checking parameters ===")
    bad = check_params_nan_inf(model)

    print("=== Checking a forward pass ===")
    check_forward_nan_inf(model, tokenizer)

    if bad:
        print("❌ Model parameters contain NaN/Inf. 这个 ckpt 已经坏了，不要再接着训了。")
    else:
        print("✅ Parameters 看起来正常，如果训练一上来就 NaN，可能是 loss / lr / optimizer 出问题。")

if __name__ == "__main__":
    main()
