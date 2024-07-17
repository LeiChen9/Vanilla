import torch
import torch.nn as nn
import torch.nn.functional as F

# define hyper params
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else "cpu"
eval_iters = 200

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

class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        logits = self.token_emb(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def gen(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLM(vocab_size).to(device)
out, loss = model(xb, yb)
print(out.shape)
print(loss)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

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