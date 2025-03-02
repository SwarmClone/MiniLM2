from transformers import ( # type: ignore
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    StoppingCriteriaList,
    StopStringCriteria
)
from time import time
from . import config

DEVICE = "cuda"

model_name = "models/transformers/ngpt/dialogue0.4b"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    trust_remote_code=True
).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompt = "你好！今天过得怎样？"
messages = [
    {"role": "system", "content": "AI是一个名叫知络的16岁女孩。AI与人类闲聊。"},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=200,
    streamer=streamer,
    stopping_criteria=StoppingCriteriaList([
        StopStringCriteria(tokenizer, "\n" * 3)
    ])
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
