
import torch 
import torch.nn  as nn 
import torch.nn.functional as F
import math 
from dataclasses import dataclass

class Attention(nn.Module):
    """"Attention as in the paper "Attention is All you need"
    does not include multihead;
    Only implements $softmax(QK^T/\sqrt {d_k}) V$, and use casual mask
    input:
        embed_dim: hidden dimension of the linear modules 
    """
    def __init__(self, embed_dim):
        super(Attention, self).__init__()
        self.fq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fv = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        q = self.fq(x)
        k = self.fk(x)
        v = self.fv(x)
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.size(-1))
        
        mask = torch.full((q.size(1), q.size(1)), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        scores = scores + mask
        scores = F.softmax(scores, dim=-1)
        
        return scores @ v

class MLP(nn.Module):
    """MLP with two layers
    input:
        embed_dim: the dimension of the input
        mlp_dim: dimension of the middle layer
    """
    def __init__(self, embed_dim, mlp_dim):
        super(MLP, self).__init__()
        self.fc = nn.Linear(embed_dim, mlp_dim, bias=False)
        self.fo = nn.Linear(mlp_dim, embed_dim, bias=False)

    def forward(self, x):
        return self.fo(F.relu(self.fc(x)))


class TransformerBlock(nn.Module):
    """Transformer block including attention and mlp"""
    def __init__(self, embed_dim, mlp_dim):
        super().__init__()
        self.self_attn = Attention(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim)
    def forward(self, x):
        """skip connection is used"""
        h = x + self.self_attn(x) 
        out = h + self.mlp(h) 
        return out


@dataclass
class SimpleLLMConfig:
    vocab_size: int = 8
    embed_dim: int = 2
    n_layers: int = 2
    mlp_dim: int = 32

class SimpleLLM(nn.Module):
    """The llm model, using SimpleLLMConfig as the configuration of hyperparameters
    Weight tied is used by setting output_proj weight equaling embedding weight
    returns:
        last_hidden_state=True, returns logits and last hidden state before output projection
        label y is None, returns logits only
        label y is not None, return logits and the loss
    """
    def __init__(self, config):
        super(SimpleLLM, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.layers = nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(TransformerBlock(config.embed_dim, config.mlp_dim))

        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.output_proj.weight.data = self.embedding.weight

    def forward(self, x , y = None, last_hidden_state = False):
        # Tokenize and embed input
        h = self.embedding(x)

        for layer in self.layers:
            h =  layer(h)
        
        logits = self.output_proj(h)
        if last_hidden_state:
            return logits, h
        if y is None:
            return logits
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
            return logits, loss


if __name__ =='__main__':
    model = SimpleLLM(SimpleLLMConfig)
    print(model)