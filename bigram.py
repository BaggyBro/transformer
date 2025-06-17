
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Hyperparameters ---
torch.manual_seed(1337)
batch_size = 64
block_size = 256
max_iters = 500
eval_interval = 10
eval_iters = 200
learning_rate = 3e-4
n_embd = 384
n_layer = 6
n_head = 6
dropout = 0.2

# --- Load Data ---
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("len:", len(text))

# --- Vocabulary and Encoding ---
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocabulary:", ''.join(chars))
print("Vocab size:", vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("Hello there!"))
print(decode(encode("Hello there!")))

# --- Data Tensor ---
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# --- Train/Val Split ---
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# --- Data Sampling ---
def get_batch(split):
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data_) - block_size, (batch_size,))
    x = torch.stack([data_[i:i+block_size] for i in ix])
    y = torch.stack([data_[i+1:i+block_size+1] for i in ix])
    return x, y

# --- Loss Estimation ---
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- Head ---
class Head(nn.Module):
    """ one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape 
        k = self.key(x)
        q = self.query(x)

        # compute attention scores 'affinities'
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,c) @ (B,T,C) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)

        #perform wewighed aggregation 
        v = self.value(x) # (B,T,C)
        out = wei @ v     # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiheadAttention(nn.Module):
    """ multiple heads of self attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transfomer block: communication followed by comparison """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: number of heads we want
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiheadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# --- Bigram Language Model ---
class BigramLanguageModel(nn.Module):
    def __init__(self ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        token_embd = self.token_embedding_table(idx)  # (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x = token_embd + pos_embd # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

# --- Initialize Model ---
model = BigramLanguageModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- Training Loop ---
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# --- Optional: Text Generation ---
print("\nGenerated text:\n")
generated = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=200)
print(decode(generated[0].tolist()))

