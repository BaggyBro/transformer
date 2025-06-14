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
val_data = data[n:]

block_size = 8
train_data[:block_size+1]
# output for this may be [10, 98 ,76 ,67]; here in the context of 10, 98 comes next; in context of 10, 98 , 76 comes next and so on

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]


torch.manual_seed(1337)
batch_size = 4 # sequences to process in parallel
block_size = 8 # maximum length for predictions 

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i:1:i+block_size] for i in ix])
    return x,y

xb,yb = get_batch('train')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]

# Bigram language model
# predicts the next character/word only based on the previos one
# one of the most simplest language model

import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token here directly reads off the logits for the next token from lookup table 
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        # idx and targets are both (B,T) tensot of integers 
        logits = self.token_embedding_table(idx) #(B,T,C)
        B,T,C = logits.shape
        logits = logits.view(B*T, C) # stretching out the array to make it 2D
        targets = targets.view(B*T) # these are needed for the pytorch functions 
        loss = F.cross_entropy(logits, targets) #loss calculation 

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step 
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get the probabilities 
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample from the distribution 
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, C)
            # append sampled index to the running sequence 
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

