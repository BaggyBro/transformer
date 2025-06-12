import torch

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("len : ", len(text))

# all the unique characters that occur in this text
chars=sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# mapping from the characters to integers
# character level tokeniser {longs sequences on word}

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("Hello there!"))
print(decode(encode("Hello there!")))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[:n]

block_size = 8
train_data[:block_size+1]
