from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel
import torch
import torch.nn.functional as F
torch.manual_seed(1234)

model_name = "alex2020/simplellm"
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


from transformers import pipeline, GenerationConfig
config = GenerationConfig(max_new_tokens=10, temperature=1.2)
generator = pipeline('text-generation', model = model, tokenizer=tokenizer, prefix="", generation_config=config)
print(generator("You do not like"))
