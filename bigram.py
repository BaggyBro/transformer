
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Hyperparameters ---
torch.manual_seed(1337)
batch_size = 32
block_size = 8
max_iters = 10000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-3

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

# --- Bigram Language Model ---
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)
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
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

# --- Initialize Model ---
model = BigramLanguageModel(vocab_size)
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

print("Final Loss:", loss.item())

# --- Optional: Text Generation ---
print("\nGenerated text:\n")
generated = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=200)
print(decode(generated[0].tolist()))

