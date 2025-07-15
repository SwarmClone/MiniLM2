from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed
)
from time import perf_counter

set_seed(42)
DEVICE = "cuda"

model_name = "models/transformers/ngpt/pretrain0.4b"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
).to(DEVICE).bfloat16()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
max_new_tokens = 1000
n = 2

print("--- KV Cache Disabled ---")

model_inputs = tokenizer(["有时候，"], return_tensors="pt").to(model.device)
speeds: list[float] = []
for i in range(n):
    t0 = perf_counter()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        use_cache=False
    )
    t1 = perf_counter()
    print(f"{(speed := max_new_tokens / (t1 - t0)):.3f} tokens/s")
    speeds.append(speed)

print(f"Average speed: {sum(speeds) / len(speeds):.3f} tokens/s")
print(f"Max speed: {max(speeds):.3f} tokens/s")
print(f"Min speed: {min(speeds):.3f} tokens/s")
print(f"Last result: {tokenizer.decode(generated_ids[0])}")

print("--- KV Cache Enabled ---")

speeds.clear()
for i in range(n):
    t0 = perf_counter()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True
    )
    t1 = perf_counter()
    print(f"{(speed := max_new_tokens / (t1 - t0)):.3f} tokens/s")
    speeds.append(speed)

print(f"Average speed: {sum(speeds) / len(speeds):.3f} tokens/s")
print(f"Max speed: {max(speeds):.3f} tokens/s")
print(f"Min speed: {min(speeds):.3f} tokens/s")
print(f"Last result: {tokenizer.decode(generated_ids[0])}")
