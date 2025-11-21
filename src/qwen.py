from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    cache_dir="/workspace/kor_med_opendataset/hg_cache"
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    cache_dir="/workspace/kor_med_opendataset/hg_cache"
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    cache_dir="/workspace/kor_med_opendataset/hg_cache"
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    cache_dir="/workspace/kor_med_opendataset/hg_cache"
)


import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

import transformers
import torch

model_id = "meta-llama/Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

pipeline("Hey how are you doing today?")



from transformers import pipeline
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it",
    device="cuda",
    torch_dtype=torch.bfloat16
)
from transformers import pipeline
import torch

pipe = pipeline("text-generation", model="google/gemma-3-1b-pt", device="cuda", torch_dtype=torch.bfloat16)
output = pipe("Eiffel tower is located in", max_new_tokens=50)
