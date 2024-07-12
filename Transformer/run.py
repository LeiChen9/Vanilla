'''
Author: LeiChen9 chenlei9691@gmail.com
Date: 2024-07-09 23:10:27
LastEditors: LeiChen9 chenlei9691@gmail.com
LastEditTime: 2024-07-13 06:22:52
FilePath: /SpeechDepDiag/Users/lei/Documents/Code/Vanilla/Transformer/run.py
Description: 

Copyright (c) 2024 by Riceball, All Rights Reserved. 
'''
import torch
import pdb 
import torch.nn as nn
import math

import torch.nn.functional as F
with open("WestWorld.txt", 'r', encoding='utf-8') as f:
    text = f.read()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get all unique position of input
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Encoding text
stoi = {ch: i for i, ch in enumerate(chars)}
itos = list(chars)
encode = lambda s: [stoi[c] for c in s] # from string to vector
decode = lambda l: "".join([itos[i] for i in l]) # from vector to string

# For test case
'''
print(encode("hello world"))
print(decode(encode("hello world")))
'''

data = torch.tensor(encode(text), dtype=torch.long )

data_len = data.size()[0]
train_len = int(data_len * 0.8)

train_data, val_data = data[:train_len], data[train_len:]

batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

x_train, y_train = get_batch("train") 

for b in range(batch_size):
    for t in range(block_size):
        context = x_train[b, :t+1]
        target = y_train[b, t]
        print("when inputs are: {}, target is {}".format(list(context), target))

config = {
    'batch_size': batch_size,
    'block_size': block_size,
    'vocab_size': vocab_size,
    'n_embed': 16,
    'dropout': 0.3,
    'n_layer': 10
}

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config['n_embed'], 4 * config['n_embed'])
        self.c_proj = nn.Linear(4*config['n_embed'], config['n_embed'])
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x):
        x = self.c_fc(x)
        x = nn.GELU(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embed'] % config['n_head'] == 0
        self.c_attn = nn.Linear(config['n_embed'], 3 * config['n_embed'])
        self.att_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])
        self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size']))
                                        .view(1, 1, config['block_size'], config['block_size']))
        
        self.n_embed = config['n_embed']
        self.n_head = config['n_head']
    
    def forward(self, x):
        B, T, C = x.size() # batch_size, seq_len, n_embed
        
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # causal self-attn: (B, nh, T, hs) x (B, nh, hs, T) = (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 * math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.att_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) = (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.resid_dropout(y)
        return y
        

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embed'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embed'])
        self.mlp = MLP(config['n_embed'])
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        
        return x
        
class ToyGPT:
    def __init__(self, config):
        self.config = config 
        self.batch_size = config['batch_size']
        self.block_size = config['block_size']
    
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config['vocab_size'], config['n_embed']),
            wpe=nn.Embedding(config['block_size'], config['n_embed']),
            dropout=nn.Dropout(config['dropout']),
            h = nn.ModuleList(Block(config) for _ in range(config['n_layer'])),
            ln_f = nn.LayerNorm(config['n_embed'])
        ))
        
        self.lm_head = nn.Linear(config['n_embed'], config['vocab_size'], bias=False)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        assert t <= self.config['block_size'], f"Cannot forward sequence of length {t}, should be less than {self.config['block_size']}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.dropout(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        # if we are given some desired targets
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=1)
        
        return logits, loss
        
        
        
    
