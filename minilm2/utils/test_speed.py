from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer
)
from time import perf_counter
from . import config

DEVICE = "cuda"

model_name = "models/transformers/ngpt/pretrain0.4b"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="bfloat16",
    trust_remote_code=True
).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model_inputs = tokenizer(["有时候，"], return_tensors="pt").to(model.device)

speeds: list[float] = []

for i in range(5):
    t0 = perf_counter()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=200
    )
    t1 = perf_counter()
    print(f"{(speed := len(generated_ids[0]) / (t1 - t0)):.3f} tokens/s")
    speeds.append(speed)

print(f"Average speed: {sum(speeds) / len(speeds):.3f} tokens/s")
print(f"Max speed: {max(speeds):.3f} tokens/s")
print(f"Min speed: {min(speeds):.3f} tokens/s")
