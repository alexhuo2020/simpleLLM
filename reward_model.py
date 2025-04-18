"""train a reward model for simulating human preference"""
import torch
import torch.nn as nn
import torch.optim as optim
from simpleLLM import SimpleLLM, SimpleLLMConfig, generate
from simpleLLM import Tokenizer, TokenizerWithSystem
import numpy as np
from post_train import resize_token_embeddings
torch.manual_seed(2345)

sft = True # whether to start from the pretrained model or used the supervised fine-tuned model

if not sft:
    tokenizer = Tokenizer()
else:
    tokenizer = TokenizerWithSystem()

# 2. Define the Reward Model (simulating human feedback)
base_model = SimpleLLM(SimpleLLMConfig)
base_model.embedding = resize_token_embeddings(base_model.embedding, 2)
base_model.output_proj = nn.Linear(SimpleLLMConfig.embed_dim, SimpleLLMConfig.vocab_size + 2, bias=False)
base_model.output_proj.weight = base_model.embedding.weight  # re-tie after replacing the embedding

class SimpleLLMForClassification(nn.Module):
    """Adding a classification head to the SimpleLLM model;
    output raw logits"""
    def __init__(self):
        super(SimpleLLMForClassification, self).__init__()
        self.model = base_model
        self.tokenizer = Tokenizer()
        self.head = nn.Sequential(
            nn.Linear(SimpleLLMConfig.embed_dim, 1),
            nn.Tanh())


    def forward(self, input_ids):
        _, outputs = self.model(input_ids, last_hidden_state=True)
        pooled = outputs[:, -1]  # last token or use mean pooling
        reward = self.head(pooled)  # (batch_size, 1)
        return reward

class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.model = SimpleLLMForClassification()
    
    def forward(self, input_ids_chosen, input_ids_rejected):
        r_chosen = self.model(input_ids_chosen)
        r_rejected = self.model(input_ids_rejected)
        return r_chosen, r_rejected
    
    def score(self, input_ids):
        with torch.no_grad():
            return self.model(input_ids).squeeze()

def generate_sample(prompt, good_resp, bad_resp):
    return {
        "prompt": prompt,
        "chosen": good_resp,
        "rejected": bad_resp
    }


if __name__ == '__main__':
    # load human preference data
    import pandas as pd 
    df = pd.read_csv("human_preference.csv")
    print(df)
    df_text = df.copy()
    df['chosen'] = df['prompt'] + ' ' + df['chosen']
    df['rejected'] = df['prompt'] + ' ' + df['rejected']

    df['chosen'] = df['chosen'].apply(tokenizer.encode)
    df['rejected'] = df['rejected'].apply(tokenizer.encode)
    df = df.drop(columns=['prompt'])  
    data = df.to_numpy()

    reward_model = RewardModel()

    # define the optimizer
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)

    # training
    losses = []
    for epoch in range(500):
        for ds in data:
            x, y = ds
            x = torch.tensor(x)
            y = torch.tensor(y)
            r_chosen, r_rejected= reward_model(x.unsqueeze(0), y.unsqueeze(0))
            loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        if epoch % 100 == 0:
            print(epoch, "loss:", loss.item())
    import matplotlib.pyplot as plt 
    plt.plot(losses)
    plt.show()

    for i in range(len(data)):
        x, y = data[i]
        x = torch.tensor(x)
        y = torch.tensor(y)
        print("score for the input", df_text.iloc[i])
        print(reward_model(x.unsqueeze(0), y.unsqueeze(0)))

    torch.save(reward_model.state_dict(), "reward_model.pt")
