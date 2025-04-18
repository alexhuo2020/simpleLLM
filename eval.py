from simpleLLM import SimpleLLM, SimpleLLMConfig, generate, Tokenizer, TokenizerWithSystem
from post_train import resize_token_embeddings


import torch
import torch.nn as nn 
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description="Evaluation script for different model types")
parser.add_argument(
    "--model_type",
    type=str,
    choices=["pretrained", "sft", "rlhf"],
    required=True,
    help="Specify the model type to evaluate: pretrained, sft, or rlhf"
)

args = parser.parse_args()

if args.model_type == "sft" or args.model_type == "rlhf":
    SimpleLLMConfig.vocab_size += 2
    model = SimpleLLM(SimpleLLMConfig)
    model.load_state_dict(torch.load("model_%s.pt"%args.model_type, weights_only=True))
    tokenizer = TokenizerWithSystem()
    X = ["human do I like coffee . system",
     "human do You like coffee . system"]

else:
    model = SimpleLLM(SimpleLLMConfig)
    model.load_state_dict(torch.load("model.pt", weights_only=True))
    tokenizer = Tokenizer()
    X = ["I like", "You like", "I do not like", "You do not like"]



for x in X:
    x = torch.tensor([tokenizer.encode(x)])
    logits = model(x)
    for _ in range(10):
        y = generate(model, x, 2)
        for yy in y:
            print(tokenizer.decode(yy))


