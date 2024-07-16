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
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode('hello, world'))
print(decode(encode('hello, world'))) 