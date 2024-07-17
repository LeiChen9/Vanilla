'''
Author: LeiChen9 chenlei9691@gmail.com
Date: 2024-07-17 10:29:34
LastEditors: LeiChen9 chenlei9691@gmail.com
LastEditTime: 2024-07-17 11:24:48
FilePath: /SpeechDepDiag/Users/lei/Documents/Code/Vanilla/Transformer/gpt.py
Description: 

Copyright (c) 2024 by Riceball, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from bigram import BigramLM

# define hyper params
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 32
head_size = 16
n_head = 3
dropout = 0.3

torch.manual_seed(3)

# open file
with open("WestWorld.txt", "r", encoding='utf-8') as f:
    text = f.read()

print("len of text: {}".format(len(text)))
print("Unique word in text: {}".format(len(set(text))))

chars = sorted(list(set(text)))
vocab_size = len(chars)

# encoding chars, can be replaced by SentencePiece of Google
# or titoken by OpenAI
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for  i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode('hello, world'))
print(decode(encode('hello, world'))) 

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

block_size = 8
batch_size = 4

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

xb, yb = get_batch('train')

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        
        self.register_buffer("trill", torch.trill(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # self-att: softmax(q @ k.t / sqrt(d)) * v 
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.trill[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHead(nn.Module):
    pass 
        

model = BigramLM(vocab_size).to(device)
out, loss = model(xb, yb)
print(out.shape)
print(loss)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch('split')
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
    
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.gen(idx, 100)[0].tolist()))