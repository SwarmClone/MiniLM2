from transformers import ( # type: ignore
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer
)
from time import time
from . import config

DEVICE = "cuda"

model_name = "models/transformers/ngpt/pretrain0.4b"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    trust_remote_code=True
).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model_inputs = tokenizer(["有时候，"], return_tensors="pt").to(model.device)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=200,
    streamer=streamer
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
