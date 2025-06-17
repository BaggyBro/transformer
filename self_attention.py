import torch 
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
B,T,C = 4, 8, 32
x = torch.randn(B,T,C)
# print(x.shape)

# we want x[b, t] = mean_{i <= t} x[b, i]
xbow = torch.zeros((B,T,C)) # bag of words
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # (t, C) -> previous tokens
        xbow[b,t] = torch.mean(xprev, 0) # averaging out the time here

print(x[0])
print(xbow[0])
# xbow[0] is averages of x[0] starting from its start 

# so a trick for this is 
a = torch.tril(torch.ones(3,3))
a = a / torch.sum(a ,1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print("a=")
print(a)
print("b=")
print(b)
print("c=")
print(c)

# similarly
wei = torch.tril(torch.ones(T,T))
wei = wei/ wei.sum(1, keepdim=True)
xbow2 = wei @ x  # (B, T, T) @ (B, T, C) --->(B, T, C)
print(torch.allclose(xbow, xbow2))


# version 3, another method
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))  # tells us how much of the previous tokens do we need to aggregate
wei = wei.masked_fill(tril == 0, float('-inf'))  # inf means we will not aggregate from those tokens {future cannot communicate with the past}
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x # aggregate the values on how interesting they find each other
print(torch.allclose(xbow, xbow3))


# version 4 -> Self attention :>
torch.manual_seed(1337)
B,T,C = 4,8,32
x = torch.randn(B,T,C)

# a single head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T ,16)
wei = q @ k.transpose(-2,-1)  # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
out = wei @ x 

v = value(x)
out = wei @ v
print(wei[0])


## Attention is a communication mechanism. Can be seen as a nodes in a directed graph looking at each other and aggregating the information with weighted sum from all nodes that point to them, with data-dependent weights
## There is no notion of space.(not like convolution). Attention simply acts over a set of vectors. This is why we need to positionally encode tokens
## Each examples across batch dimension is of course processed completely independently and never talk to each other.
## In an encoder attention block just delete the single line that does masking with tril, allowing all the tokens to communicate. Here is called a decoder attention block because it has triangular masking and is usually used in autoregressive settings.


## Self attention -> key query values all come from x 
