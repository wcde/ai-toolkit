import torch
from transformers import AutoTokenizer

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-4B",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)

a = pipe.tokenize(
    "1",
    padding="max_length",
    max_length=pipe.model_max_length,
    truncation=True,
    return_tensors="pt",
    )

print(a)
