import torch
import torch.nn as nn
import torch.nn.functional as F
from simpleLLM import SimpleLLM, SimpleLLMConfig
from reward_model import RewardModel
from post_train import resize_token_embeddings, Tokenizer
from simpleLLM import generate
import numpy as np
# whether to use the supervised fine tuned model or not
sft = True

# 1. Define the Policy Model (pretrained LLM w/o fine tune)
model = SimpleLLM(SimpleLLMConfig)
if not sft:
    model.load_state_dict(torch.load("model.pt", weights_only=True))
    model.embedding = resize_token_embeddings(model.embedding, 2)
    model.output_proj = nn.Linear(SimpleLLMConfig.embed_dim, SimpleLLMConfig.vocab_size + 2, bias=False)
    model.output_proj.weight = model.embedding.weight  # re-tie after replacing the embedding
else:
    model.embedding = resize_token_embeddings(model.embedding, 2)
    model.output_proj = nn.Linear(SimpleLLMConfig.embed_dim, SimpleLLMConfig.vocab_size + 2, bias=False)
    model.output_proj.weight = model.embedding.weight  # re-tie after replacing the embedding
    model.load_state_dict(torch.load("model_sft.pt", weights_only=True))
tokenizer = Tokenizer()

class PolicyModel(nn.Module):
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.model = model
        self.tokenizer = Tokenizer()
        self.critic_head = nn.Linear(SimpleLLMConfig.embed_dim, 1)


    def forward(self, input_ids):
        logits = self.model(input_ids)
        h = self.model(input_ids, last_hidden_state=True)
        value = self.critic_head(h)
        return logits, value.squeeze(-1) # batchsize, seq_len, 1

    def generate(self, prompt, max_length=1):
        input_ids = torch.tensor([self.tokenizer.encode(prompt)])
        output_ids = generate(self.model, input_ids, max_length=max_length)
        generated_text = self.tokenizer.decode(output_ids[0])
        return generated_text


# 2. Define the Reward Model, see reward_model.py for how to train this model 
reward_model = RewardModel()
reward_model.load_state_dict(torch.load("reward_model.pt", weights_only=True))

# 3. PPO Helper Functions (Generalized Advantage Estimation)
# def compute_advantages(rewards, values, gamma=0.99, lambda_=0.95):
#     print(rewards.shape, values.shape)
#     advantages, gae = [], 0
#     values = torch.cat((values, torch.tensor([0])))
#     for t in reversed(range(len(rewards))):
#         delta = rewards[t] + gamma * values[t + 1] - values[t]
#         gae = delta + gamma * lambda_ * gae
#         advantages.insert(0, gae)
#     returns = torch.stack([adv + val for adv, val in zip(advantages, values[:-1])])
#     return torch.stack(advantages), returns
def compute_advantages(rewards, values, gamma=0.99, lambda_=0.95):
    """
    Compute advantages using Generalized Advantage Estimation (GAE) for a batch of rewards and values.
    
    rewards: Tensor of shape (batch_size, seq_len) containing rewards for each timestep
    values: Tensor of shape (batch_size, seq_len) containing value predictions for each timestep
    gamma: Discount factor (usually close to 1, e.g., 0.99)
    lambda_: GAE parameter (usually close to 1, e.g., 0.95)
    
    Returns:
    advantages: Tensor of shape (batch_size, seq_len) containing the advantage for each timestep
    returns: Tensor of shape (batch_size, seq_len) containing the return for each timestep
    """
    batch_size, seq_len = rewards.size()

    # Add a column of zeros at the end (next timestep's value is 0)
    values = torch.cat((values, torch.zeros(batch_size, 1)), dim=1)  # shape (batch_size, seq_len + 1)

    advantages = torch.zeros_like(rewards)  # Initialize advantages tensor (same shape as rewards)
    returns = torch.zeros_like(rewards)  # Initialize returns tensor (same shape as rewards)

    # Iterate over time steps in reverse order
    for t in reversed(range(seq_len)):
        delta = rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
        advantages[:, t] = delta + gamma * lambda_ * advantages[:, t + 1] if t < seq_len - 1 else delta
        returns[:, t] = advantages[:, t] + values[:, t]
    return advantages, returns

def get_logprob(logits, input_ids, response_mask):
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)
    response_token_log_probs = token_log_probs * response_mask  # (batch, seq_len)
    return response_token_log_probs

# 4. PPO Objective with Clipping
def ppo_loss(old_log_probs, new_log_probs, advantages, epsilon=0.1):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    loss = - torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss


# 5. PPO Training Loop
def ppo_train(policy_model, reward_model, ref_model, optimizer, prompts, num_iters=10, num_epochs=50, max_length=1):
    for iter in range(num_iters):
        # Simulate an environment loop
        rewards = []
        log_probs = []
        values = []
        advantages = []
        old_log_probs = []
        responses = []
        masks = []

        for prompt in prompts:
            with torch.no_grad():
                generated_text = policy_model.generate(prompt, max_length=2)
                # Compute the reward for generated text
                input_ids = torch.tensor([tokenizer.encode(generated_text)])
                responses.append(input_ids)
                reward = reward_model.score(input_ids)
                logits_old, _ = ref_model(input_ids)
                _, value = policy_model(input_ids)

                # mask prompt
                response_mask = torch.zeros_like(input_ids)
                prompt_len = len(tokenizer.encode(prompt))
                response_mask[:, prompt_len:] = 1
                logprob_old = get_logprob(logits_old, input_ids, response_mask)
                logprob_old = logprob_old[:, prompt_len:]

                reward_per_token = torch.zeros_like(input_ids, dtype=torch.float)
                reward_per_token[0, -1] = reward  # place reward on last token only

                value = value[:, prompt_len:]
                reward_per_token = reward_per_token[:, prompt_len:]



            rewards.append(reward_per_token)
            old_log_probs.append(logprob_old)  # Simulated log probabilities
            values.append(value)
            masks.append(response_mask)
        

        rewards = torch.stack(rewards).squeeze(1)
        values = torch.stack(values).squeeze(1)
        
        # Compute advantages
        advantages, returns = compute_advantages(rewards, values)

        for epoch in range(num_epochs):
            log_probs = []
            value_preds = []
            for i in range(len(responses)):
                input_ids = responses[i]
                logits_new, value = policy_model(input_ids)
                response_mask = torch.zeros_like(input_ids)
                response_mask[:, prompt_len:] = 1  
                logprob_new = get_logprob(logits_new, input_ids, response_mask)
                logprob_new = logprob_new[:, prompt_len:]
                log_probs.append(logprob_new)
                value = value[:, prompt_len:]
                # value = value[0,-1].squeeze()
                # value_preds.append(value)
                # value_preds = torch.stack(value_preds)
            # Calculate loss using PPO objective with the clipped advantage
                value_loss = F.mse_loss(value,returns[i])
                kl_loss = (logprob_new - old_log_probs[i]).mean()
                loss = ppo_loss(old_log_probs[i], logprob_new, advantages[i]) + 0.5 * value_loss + 0.1 * kl_loss

                # Backpropagate and update policy model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        ref_model.load_state_dict(policy_model.state_dict())

# 6. Initialize models and optimizer
policy_model = PolicyModel()
reward_model = RewardModel()
ref_model = PolicyModel()
# ref_model.eval()
reward_model.eval()
optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-4)

# Train the model
ppo_train(policy_model, reward_model, ref_model, optimizer, prompts=["do I like coffee . system"])#, "do You like tea . system"])

# evaluation
X = ["human do I like coffee . system"]#,
    #  "human do You like tea . system",
    #  "human do You like coffee . system"]
for x in X:
    xx = torch.tensor([policy_model.tokenizer.encode(x)])

    logits = policy_model.model(xx)
    print(F.softmax(logits, -1))    
    for _ in range(20):

        generated_text = policy_model.generate(x)
        print(generated_text)


torch.save(policy_model.model.state_dict(), "model_rlhf.pt")
