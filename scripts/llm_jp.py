import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-13b-instruct3")
model = AutoModelForCausalLM.from_pretrained("llm-jp/llm-jp-3-13b-instruct3", 
    device_map="cuda", torch_dtype=torch.bfloat16)
model.eval()
system_example = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"
instruction_example = "アニメ「鬼滅の刃」の主人公は誰ですか？"

prompt = f"{system_example}\n\n### 指示:\n{instruction_example}\n\n### 応答:\n"

chat = [
    {"role": "system", "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"},
    {"role": "user", "content": prompt},
]

start = time.time()
tokenized_input = tokenizer.apply_chat_template(chat, add_generation_prompt=True, 
    tokenize=True, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        tokenized_input,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.05,
    )[0]
stop = time.time()

print(f"Time taken: {(stop - start) * 1e3}ms")
print(tokenizer.decode(output))
