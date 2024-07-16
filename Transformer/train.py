import torch
import torch.nn as nn
import torch.nn.functional as F

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
    return x, y

xb, yb = get_batch('train')

class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets):
        logits = self.token_emb(idx)

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)

        loss = F.cross_entropy(logits, targets)

        return logits, loss

m = BigramLM(vocab_size)
out, loss = m(xb, yb)
print(out.shape)
print(loss)