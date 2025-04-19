import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

model_name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)
chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
set_seed(123)

system_example = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"
instruction_example = "アニメ「鬼滅の刃」の主人公は誰ですか？"
prompt = f"{system_example}\n\n### 指示:\n{instruction_example}\n\n### 応答:\n"


user_input = [{"role": "user", "content": prompt}]

# Warmup run because the first run seems to require cache warmup.
start = time.time()
responses = chat_pipeline(
    user_input,
    max_length=200,
    do_sample=True,
    num_return_sequences=1,
)
end = time.time()
print(f"time: {((end - start) * 1e3) / 10}ms")

for i, response in enumerate(responses, 1):
    print(f"Response {i}: {response['generated_text']}")