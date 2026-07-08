##this is my code.
#%%
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# %%
print("length of dataset in characters: ", len(text))


# %%
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(''.join(chars))
print(vocab_size)


# %%
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hii there"))
print(decode(encode("hii there")))


# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# device is defined here, EARLY, before anything (like get_batch) needs to use it.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)

data = torch.tensor(encode(text), dtype=torch.long)

print(data.shape, data.dtype)
print(data[:1000])


# %%
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# %%
block_size = 8
train_data[:block_size+1]

# %%
x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")


# %%
torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")


# %%
print(xb)

# %%
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B,T,C)

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
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# NOTE: this bigram model is moved to device too, so it matches xb/yb which are now on device
m = BigramLanguageModel(vocab_size).to(device)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

# %%
print("expected loss at initialization: ", math.log(vocab_size))
print("actual loss:                     ", loss.item())


# %%
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx=idx, max_new_tokens=100)[0].tolist()))

# %%
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# %%
batch_size = 32

for steps in range(100):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())


# %%
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx=idx, max_new_tokens=500)[0].tolist()))

# %%
torch.manual_seed(42)

a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)

b = torch.randint(0, 10, (3, 2)).float()
c = a @ b

print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)


# %%
torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
x.shape

# %%
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, 0)

# %%
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)

xbow2 = wei @ x
torch.allclose(xbow, xbow2)

# %%
tril = torch.tril(torch.ones(T, T))

wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

xbow3 = wei @ x
torch.allclose(xbow, xbow3)

# %%
torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

head_size = 16

key   = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)
q = query(x)

wei = q @ k.transpose(-2, -1)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v

out.shape


# %%
k_demo = torch.randn(B, T, head_size)
q_demo = torch.randn(B, T, head_size)
wei_demo = q_demo @ k_demo.transpose(-2, -1)

print("var(k):  ", k_demo.var().item())
print("var(q):  ", q_demo.var().item())
print("var(wei):", wei_demo.var().item())


# %%
example_logits = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])

print("modest-scale logits  -> softmax:", torch.softmax(example_logits, dim=-1))
print("8x larger logits     -> softmax:", torch.softmax(example_logits * 8, dim=-1))


# %%
wei_scaled = (q_demo @ k_demo.transpose(-2, -1)) * head_size**-0.5

print("var(wei), unscaled:", wei_demo.var().item())
print("var(wei), scaled:  ", wei_scaled.var().item())

# %%
wei = q @ k.transpose(-2, -1) * head_size**-0.5
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

out = wei @ v
out.shape

# %%
class LayerNorm1d:

  def __init__(self, dim, eps=1e-5, momentum=0.1):
      self.eps = eps
      self.gamma = torch.ones(dim)
      self.beta = torch.zeros(dim)

  def __call__(self, x):
      xmean = x.mean(1, keepdim=True)
      xvar = x.var(1, keepdim=True)
      xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
      self.out = self.gamma * xhat + self.beta
      return self.out

  def parameters(self):
      return [self.gamma, self.beta]

torch.manual_seed(1337)
module = LayerNorm1d(100)
x = torch.randn(32, 100)
x = module(x)
x.shape

# %%
x[:,0].mean(), x[:,0].std()
# %%
x[0,:].mean(), x[0,:].std()
# %%
# NOTE: device is NOT redefined here anymore -- it was already set once, at the top.
block_size = 32
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

# %%
class Head(nn.Module):
    ''' one head of self-attention '''

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # FIX: scale by head_size (the actual key/query dim), not C (which is n_embd here)
        wei = q @ k.transpose(-2,-1) * (self.key.out_features ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

# %%
head_size = n_embd // n_head
dummy_x = torch.randn(2, 10, n_embd)
test_head = Head(head_size)
print(test_head(dummy_x).shape)

# %%
class MultiHeadAttention(nn.Module):
    ''' multiple heads of self-attention in parallel '''

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# %%
test_mha = MultiHeadAttention(n_head, head_size)
print(test_mha(dummy_x).shape)


# %%
class FeedFoward(nn.Module):
    ''' a simple linear layer followed by a non-linearity '''

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# %%
test_ffwd = FeedFoward(n_embd)
print(test_ffwd(dummy_x).shape)


# %%
class Block(nn.Module):
    ''' Transformer block: communication followed by computation '''

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# %%
test_blocks = nn.Sequential(Block(n_embd, n_head), Block(n_embd, n_head))
print(test_blocks(dummy_x).shape)


# %%
position_embedding_table = nn.Embedding(block_size, n_embd)

T_demo = 10
pos_emb_demo = position_embedding_table(torch.arange(T_demo))
print(pos_emb_demo.shape)


# %%
tok_emb_demo = torch.randn(2, T_demo, n_embd)
combined = tok_emb_demo + pos_emb_demo
print(combined.shape)

#%%
batch_size = 16
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200

# %%
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

# %%
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
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
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# %%
torch.manual_seed(1337)

model = GPTLanguageModel()
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# %%
for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# %%
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# %%
def generate_text(prompt="", num_words=50):
    ''' feed the model a starting prompt and a target word count '''
    model.eval()
    ids = [stoi[c] for c in prompt if c in stoi]
    if not ids:
        ids = [0]
    context = torch.tensor([ids], dtype=torch.long, device=device)

    # generate more characters than we probably need, then trim to exact word count
    approx_chars_needed = num_words * 6  # ~5 chars/word + space, with buffer
    out_ids = m.generate(context, max_new_tokens=approx_chars_needed)[0].tolist()
    full_text = decode(out_ids)

    # trim down to exactly num_words words
    words = full_text.split()
    trimmed = ' '.join(words[:num_words])

    model.train()
    return trimmed


# example usage:
if __name__ == "__main__":
    print("Type a starting prompt, then how many words to generate.\n")
    while True:
        try:
            prompt = input("Starting words: ")
            if prompt.strip().lower() in ("", "quit", "exit"):
                break
            num_words = int(input("How many words to generate: "))
        except (EOFError, KeyboardInterrupt):
            break
        print()
        print(generate_text(prompt, num_words))
        print()
# %%
