from transformers import ( # type: ignore
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    StoppingCriteriaList,
    StopStringCriteria
)
from ..llm import model as llm_model
from time import time
from . import config

DEVICE = "cuda"

model_name = "models/transformers/ngpt/sft0.4b"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto"
).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "请向我介绍什么是大语言模型。"
messages = [
    {"role": "system", "content": "AI是一个名叫MiniLM2的小型语言模型。AI是人类的助手，会回答用户的问题并遵守用户的指令。"},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    stopping_criteria=StoppingCriteriaList([
        StopStringCriteria(tokenizer, "\n" * 3)
    ])
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
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
