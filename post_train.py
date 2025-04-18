

"""SFT on SimpleLLM, using full parameter fine tuning"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from simpleLLM import SimpleLLM, SimpleLLMConfig, generate
from dataclasses import dataclass
torch.manual_seed(1234)

@dataclass
class TrainConfig:
    lr: float = 1e-3
    max_epochs: int = 5000

# use this or import from simpleLLM using 
# from simpleLLM import TokenizerWithSystem
class TokenizerWithSystem:
    def __init__(self):
        self.word_to_id = {"I":0, "You":1, "like": 2, "do": 3, "not": 4, "coffee":5,"tea":6,".": 7, "human": 8, "system": 9}
        self.id_to_word = {v:k for k,v in self.word_to_id.items()}
    def encode(self, x):
        return [self.word_to_id[c] for c in x.split()]
    def decode(self, x):
        return ' '.join([self.id_to_word[c.item()] for c in x])

def resize_token_embeddings(embed, num_new_tokens):
    """Adjust embedding layer for new introduced tokens, keeeping the old embedding weight
    for existing tokens
    input:
        embed: nn.Embedding type layer
        num_new_tokens: number of added tokens
    """
    old_vocab_size, embedding_dim = embed.weight.shape
    new_vocab_size = old_vocab_size + num_new_tokens

    # Create new embedding layer
    new_embed = nn.Embedding(new_vocab_size, embedding_dim)

    # Copy weights from the old embedding
    new_embed.weight.data[:old_vocab_size] = embed.weight.data

    # Initialize the new embeddings (e.g., normal)
    # nn.init.normal_(new_embed.weight.data[old_vocab_size:])

    return new_embed



if __name__ == '__main__':
    # preparing data
    data = ["human do I like coffee . system do .",
            "human do You like coffee . system not .", 
            "human do I like coffee . system do like",
            "human do You like coffee . system not like"]
    answer_length = 2
    tokenizer = TokenizerWithSystem()

    train_ids = []
    for s in data:
        train_ids.append(tokenizer.encode(s))

    x = [torch.tensor(d[:-1]) for d in train_ids]

    y = []
    for d in train_ids:
        labels = torch.tensor(d[1:])  # original target, shifted
        mask = torch.full_like(labels, -1)  # initialize all -1
        mask[-answer_length:] = labels[-answer_length:]  # apply only to assistant's response
        y.append(mask)

    train_ids = list(zip(x,y))

    # define the model and load from the trained weight
    model = SimpleLLM(SimpleLLMConfig)
    model.load_state_dict(torch.load("model.pt", weights_only=True))

    # adjust the embedding and final projection layer for the added tokens
    model.embedding = resize_token_embeddings(model.embedding, 2)
    model.output_proj = nn.Linear(SimpleLLMConfig.embed_dim, SimpleLLMConfig.vocab_size + 2, bias=False)
    model.output_proj.weight = model.embedding.weight  # re-tie after replacing the embedding


    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainConfig.lr)

    # training
    losses = []
    for epoch in range(TrainConfig.max_epochs):
        for x, y in train_ids:
            _, loss = model(x.unsqueeze(0), y.unsqueeze(0))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        if epoch % 100 == 0:
            print(epoch, "training loss:", loss.item())

    # save the result
    torch.save(model.state_dict(), "model_sft.pt")

    # evaluation 
    import matplotlib.pyplot as plt 
    plt.plot(losses)
    plt.savefig("loss.png")
    # evaluation
    X = ["human do I like coffee . system", "human do You like coffee . system",
         "human do I like tea . system",
         "human do You like tea . system"]
    for x in X:
        x = torch.tensor([tokenizer.encode(x)])

        logits = model(x)
        print(F.softmax(logits, -1))
        for _ in range(10):
            y = generate(model, x, 1)
            for yy in y:
                print(tokenizer.decode(yy))

