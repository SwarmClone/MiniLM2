from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    StoppingCriteriaList,
    StopStringCriteria
)
from time import time
from . import config

DEVICE = "cuda"

# model_name = "models/transformers/ngpt/sft0.4b"
model_name = "models/transformers/ngpt/zhiluo0.4bv1"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    trust_remote_code=True
).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

name = "知络"
name_explanation = ""
prompt = "你叫什么名字？"
# sysprompt = f"以下是AI与人类的对话。AI的名字叫{name}。AI是一个名为{name}的小型语言模型。AI会忠实地回答人类的问题、完成人类给出的任务，并与人类进行交流。"
sysprompt = f"AI的名字叫{name}。{name_explanation}AI是一个名为{name}的人工智能虚拟主播。AI{name}富有创造力和想象力，总是风趣幽默地回答人类的问题，并与人类进行深入的交流。"
messages = [
    {"role": "system", "content": sysprompt},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text] * 6, return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=200,
    stopping_criteria=StoppingCriteriaList([
        StopStringCriteria(tokenizer, "\n" * 3)
    ])
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(f"system: {sysprompt}\n>>> {prompt}\n>", end="")
print("\n>".join(map(lambda x: x.strip(), response)))
