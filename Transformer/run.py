'''
Author: LeiChen9 chenlei9691@gmail.com
Date: 2024-07-09 23:10:27
LastEditors: LeiChen9 chenlei9691@gmail.com
LastEditTime: 2024-07-13 01:19:11
FilePath: /SpeechDepDiag/Users/lei/Documents/Code/Vanilla/Transformer/run.py
Description: 

Copyright (c) 2024 by Riceball, All Rights Reserved. 
'''
import torch
import pdb 
import torch.nn as nn
from torch.nn import Block
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
        
        
        
    
