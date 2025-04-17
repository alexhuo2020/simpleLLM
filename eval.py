from simpleLLM import SimpleLLM, SimpleLLMConfig, generate
from post_train import resize_token_embeddings
from post_train import Tokenizer
# from tokenizer import Tokenizer

import torch
import torch.nn as nn 
import torch.nn.functional as F
# from generate import generate
model = SimpleLLM(SimpleLLMConfig)

model.embedding = resize_token_embeddings(model.embedding, 2)
model.output_proj = nn.Linear(SimpleLLMConfig.embed_dim, SimpleLLMConfig.vocab_size + 2, bias=False)
model.output_proj.weight = model.embedding.weight  # re-tie after replacing the embedding
model.load_state_dict(torch.load("model_rlhf.pt", weights_only=True))
tokenizer = Tokenizer()

# X = ["I like", "You like", "You do not like", "I do not like", "like"]
X = ["human do I like coffee . system"]#,
    #  "human do You like tea . system",
    #  "human do You like coffee . system"]

for x in X:
    x = torch.tensor([tokenizer.encode(x)])

    logits = model(x)
    print(F.softmax(logits, -1))    
    for _ in range(10):
        y = generate(model, x, 2)
        for yy in y:
            print(tokenizer.decode(yy))


