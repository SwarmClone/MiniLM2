from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
)

DEVICE = "cuda"

model_name = "models/transformers/ngpt/sft0.4b"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    trust_remote_code=True
).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompt = "请向我介绍什么是大语言模型。"
messages = [
    {"role": "system", "content": "AI是一个名叫MiniLM2的小型语言模型。AI是人类的助手，会回答用户的问题并遵守用户的指令。"},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
streamer = TextStreamer(tokenizer, skip_prompt=True)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768,
    streamer=streamer,
    use_cache=True,
)