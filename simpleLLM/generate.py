import torch
import torch.nn.functional as F

@torch.inference_mode()
def generate(model, input_tokens, max_length=1):
    """inference function for large language models
    input:
        model: a pytorch llm model
        input_tokens: batch_size x seq_len of long int type
        max_length: length of the generated sequence
    return:
        the full tokens (including input tokens)
    """
    for _ in range(max_length):
        logits = model(input_tokens)
        probs = F.softmax(logits, -1)
        next_token_probs = probs[:,-1]
        next_token = torch.multinomial(next_token_probs, num_samples=1)
        input_tokens = torch.cat([input_tokens, next_token], dim=-1)
    return input_tokens

if __name__ == '__main__':
    from .model import SimpleLLM, SimpleLLMConfig
    from .tokenizer import Tokenizer
    tokenizer = Tokenizer()
    model = SimpleLLM(SimpleLLMConfig)
    x = "I like"
    x = torch.tensor([tokenizer.encode(x)])
    y = generate(model, x, 2)
    for yy in y:
        print(tokenizer.decode(yy))

    
    input_tokens = []
    x = ["I like", "You like"]

    for xx in x: 
        input_tokens.append(tokenizer.encode(xx))
    input_tokens = torch.tensor(input_tokens)
    
    
    y = generate(model, input_tokens, 2)
    for yy in y:
        print(tokenizer.decode(yy))


