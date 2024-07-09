'''
Author: LeiChen9 chenlei9691@gmail.com
Date: 2024-07-09 23:10:27
LastEditors: LeiChen9 chenlei9691@gmail.com
LastEditTime: 2024-07-09 23:23:41
FilePath: /SpeechDepDiag/Users/lei/Documents/Code/Vanilla/Transformer/run.py
Description: 

Copyright (c) 2024 by Riceball, All Rights Reserved. 
'''
import torch
with open("WestWorld.txt", 'r', encoding='utf-8') as f:
    text = f.read()

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