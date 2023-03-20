# from transformers import AutoConfig
# from modeling_simplellm import SimpleLLMModel, SimpleLLMConfig, SimpleLLMForCausalLM
# import torch


from tokenization_simplellm import SimpleLLMTokenizer

SPECIAL_TOKENS = ["<|endoftext|>"]

tokenizer = SimpleLLMTokenizer(special_tokens=SPECIAL_TOKENS)
tokenizer.push_to_hub("alex2020/simplellm", commit_message="Added simplellm tokenizer")


text = "I like coffee ."
tokens = tokenizer.tokenize(text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)
print("Input IDs:", input_ids)
print("Decoded:", tokenizer.decode(input_ids))

print(tokenizer.vocab_size)
# config = SimpleLLMConfig(vocab_size=tokenizer.vocab_size, embed_dim=8, n_layers=2, mlp_dim=32)
# model = SimpleLLMForCausalLM(config)
inputs = tokenizer(text)
print(inputs)
# x = torch.randint(0, 100, (1, 10))  # batch size 1, sequence length 10
# y = torch.randint(0, 100, (1, 10))

# output = model(x, labels=y)
# print(output["loss"], output["logits"].shape)
tokenizer.save_pretrained("./tokenize")
print(tokenizer.all_special_tokens)
# Tokenization function with labels
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        padding=True,
        truncation=True,
        max_length=6,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens
tokenizer.pad_token = tokenizer.all_special_tokens[0]
from datasets import load_dataset
train_data = load_dataset("alex2020/SimpleDataset")# Tokenize and add labels
train_dataset = train_data.map(tokenize, batched=True)

