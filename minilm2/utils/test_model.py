from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StopStringCriteria # type: ignore
from ..llm import model as llm_model
from time import time

DEVICE = "cuda"

model_name = "models/transformers/ngpt/sft0.4b"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto"
).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("models/tokenizers/tokenizer32k")

prompt = "向我介绍一下大语言模型。"
messages = [
    {"role": "system", "content": "你是一个名叫MiniLM2的小型语言模型。你是一个助手。"},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=100,
    stopping_criteria=StoppingCriteriaList([StopStringCriteria(tokenizer, ["\n\n\n"])])
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
