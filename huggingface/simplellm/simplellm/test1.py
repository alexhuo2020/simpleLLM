from transformers import AutoModel, AutoTokenizer, AutoConfig

# Replace with your Hugging Face model identifier
model_name = "alex2020/simplellm"

# Load the model and tokenizer
#config = AutoConfig.from_pretrained(model_name)
#model = AutoModel.from_config(config)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,special_tokens=["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
)
print(tokenizer.vocab_size)
text = "I like coffee . <|endoftext|>"
print(tokenizer(text))
