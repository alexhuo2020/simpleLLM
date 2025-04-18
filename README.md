
# SimpleLLM: a simple large language model for education purpose

<div align="center">
  <img src="https://github.com/alexhuo2020/simpleLLM/blob/main/logo.png" width="40%" alt="SimpleLLM" />
</div>


## 1. Introduction
SimpleLLM is a simple large language model that aims to present the whole cycle of training, fine-tuning and reinforcement learning from scratch on a simple dataset (a few sentences). It requires only CPU to run the model inference and training. It is primarily aimed to illustrate the process to develop an LLM in the most simple form.

## 2. Data
We train the model on the sentences

- I like coffee .
- I like tea .
- You like tea .
- You do not like coffee .

A simple tokenizer is used that maps the words to integers. 

## 3. Model
We use the Attention (with only one head) and MLP (a simple two layer fully connected network) as the building blocks for the transformer. The model lacks the normalization layer, multi-head support, mixture of experts, positional embeddings, etc. 

To explore the transformer structure, one can run the jupyter notebook ![playground.ipynb](./playground.ipynb)

## 4. Pretraining
Pretraining is done on the data above. Run 
```
python train.py
```
to train the model and obtain the model weight "model.pt". 

## 5. Supervised Fine-Tuning (SFT)
We use the following data for sft:

    data = ["human do I like coffee . system do .",
            "human do You like coffee . system not .", 
            "human do I like coffee . system do like",
            "human do You like coffee . system not like"]

The fine-tuned model will be a chat model and can answer the question correctly. Run 
```
python post_train.py
```
to fine tune the model and obtain the fined model weight "model_sft.pt".

## 6. RLHF (Reinforcement learning from human feedback)
We will train the model to prefer answer the question 

"do I like coffee"

with 

"do ." rather than "do like".

And 

"do You like coffee"

with 

"not ." rather than "not like".

Run 
```
python rlhf.py
```
to do RLHF and obtain the model weight "model_rlhf.pt".

## 7. Evaluation
To evaluate the model, one can run 
```
python eval.py --model_type pretrained/sft/rlhf
```


