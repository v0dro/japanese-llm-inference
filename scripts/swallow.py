from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import time
import os

os.environ['VLLM_USE_V1'] = '0'

model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(
    model=model_name,
    tensor_parallel_size=1,
)
sampling_params = SamplingParams(
    temperature=0.6, top_p=0.9, max_tokens=512, stop="<|eot_id|>"
)

system_example = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"
instruction_example = "アニメ「鬼滅の刃」の主人公は誰ですか？"
prompt = f"{system_example}\n\n### 指示:\n{instruction_example}\n\n### 応答:\n"

start_time = time.time()
message = [
    {"role": "system", "content": "あなたは誠実で優秀な日本人のアシスタントです。"},
    {
        "role": "user",
        "content": prompt,
    },
]
prompt = tokenizer.apply_chat_template(
    message, tokenize=False, add_generation_prompt=True
)
output = llm.generate(prompt, sampling_params)
end_time = time.time()

print(f"Time taken: {(end_time - start_time) * 1e3}ms")
print(output[0].outputs[0].text)
