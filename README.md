# DRT-o1

<p align="center">
🤗 <a href="https://huggingface.co/Krystalan/DRT-o1-7B">DRT-o1-7B</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/Krystalan/DRT-o1-14B">DRT-o1-14B</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2412.17498">Paper</a>

</p>

This repository contains the resources for our paper ["DRT-o1: Optimized Deep Reasoning Translation via Long Chain-of-Thought"](https://arxiv.org/abs/2412.17498)


### Updates:
- *2024.12.24*: We released [our paper](https://arxiv.org/abs/2412.17498). Check it out!
- *2024.12.23*: We released our model checkpoints. 🤗 <a href="https://huggingface.co/Krystalan/DRT-o1-7B">DRT-o1-7B</a> and 🤗 <a href="https://huggingface.co/Krystalan/DRT-o1-14B">DRT-o1-14B</a>.


## Introduction

![](./images/multi-agent-framework.png)



In this work, we introduce DRT-o1, an attempt to bring the success of long thought reasoning to neural machine translation (MT). To this end,
- 🌟 We mine English sentences with similes or metaphors from existing literature books, which are suitable for translation via long thought.
- 🌟 We propose a designed multi-agent framework with three agents (i.e., a translator, an advisor and an evaluator) to synthesize the MT samples with long thought. There are 22,264 synthesized samples in total.
- 🌟 We train DRT-o1-7B and DRT-o1-14B using Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct as backbones.


![](./images/data_case.png)


## Quickstart

### ⛷️ Huggingface Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Krystalan/DRT-o1-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Please translate the following text from English to Chinese:\nThe mother, with her feet propped up on a stool, seemed to be trying to get to the bottom of that answer, whose feminine profundity had struck her all of a heap."
messages = [
    {"role": "system", "content": "You are a philosopher skilled in deep thinking, accustomed to exploring complex problems with profound insight."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### ⛷️ vllm

Deploying LLMs:
```bash
python3 -m vllm.entrypoints.openai.api_server --model [model_ckpt] --served-model-name [model_name]
```

Calling LLMs:
```python
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model=[model_name],
    messages=[
        {"role": "system", "content": "You are a philosopher skilled in deep thinking, accustomed to exploring complex problems with profound insight."},
        {"role": "user", "content": "Please translate the following text from English to Chinese:\nThe mother, with her feet propped up on a stool, seemed to be trying to get to the bottom of that answer, whose feminine profundity had struck her all of a heap."},
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=2048,
    extra_body={
        "repetition_penalty": 1.05,
    },
)
print("Chat response:", chat_response)
```


## License
This work is licensed under cc-by-nc-sa-4.0

