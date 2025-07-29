from vllm import LLM, SamplingParams

# 你要加载的模型
model_name = "google/gemma-2-9b-it"  # 需要确保你有访问权限

# 初始化 vLLM 模型
llm = LLM(model=model_name)

# 定义测试 prompt
prompts = [
    "Hello! Can you tell me a fun fact about space?",
    "写一句励志的中文名言。"
]

# 设置采样参数
sampling_params = SamplingParams(temperature=0.7, max_tokens=64)

# 运行推理
outputs = llm.generate(prompts, sampling_params)

# 打印结果
for i, output in enumerate(outputs):
    print(f"Prompt {i+1}: {prompts[i]}")
    print(f"Output: {output.outputs[0].text}")
    print("="*50)
