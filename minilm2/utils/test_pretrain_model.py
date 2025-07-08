from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer
)
from time import time
from . import config

DEVICE = "cuda"

model_name = "models/transformers/ngpt/checkpoint_5400_3.6507.pt"
# model_name = "models/transformers/ngpt/pretrain0.4b"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    trust_remote_code=True
).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model_inputs = tokenizer(["An incrementing sequence of numbers: 1, 2, 3, "], return_tensors="pt").to(model.device)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=200,
    streamer=streamer,
    do_sample=False,
    temperature=0
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
