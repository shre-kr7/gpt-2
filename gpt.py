#%%
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # avoid cp1252 crashes on Windows consoles

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#%%
print("length of dataset in characters: ", len(text))

# %%
print(text[:1000])


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
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # defined early so get_batch can use it below
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
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data

   
    ix = torch.randint(len(data) - block_size, (batch_size,))

    
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)   # move batch to GPU if available -- was missing, silently only worked on CPU
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')


for b in range(batch_size):   # batch dimension
    for t in range(block_size):   # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")


# %%
print(xb)

# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            # At generation time we don't have a "correct answer" to compare against,
            # so there's nothing to compute a loss against.
            loss = None
        else:
            B, T, C = logits.shape
           
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T): a batch of token-index sequences we want to extend.
        for _ in range(max_new_tokens):
            # get the predictions for every position, given the sequence so far
            logits, loss = self(idx)
            # we only care about the prediction for the NEXT token, which comes from the
            # logits at the LAST time step. Shape (B, T, C) -> (B, C).
            logits = logits[:, -1, :]
            # convert raw logits into a valid probability distribution over the vocabulary
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample one token index per batch row from that distribution (NOT argmax —
            # sampling keeps generation varied instead of always picking the single most
            # likely character, which tends to produce repetitive loops)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # stick the newly generated token onto the end of the running sequence, and repeat
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)


# %%
import math
print("expected loss at initialization: ", math.log(vocab_size))
print("actual loss:                     ", loss.item())

# %%
idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx=idx, max_new_tokens=100)[0].tolist()))

# %%
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# %%
batch_size = 32   # we can use a bigger batch now that we're training "for real"

for steps in range(100):   # increase number of steps for good results...

    # sample a fresh random batch of data every step
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)  # clear gradients from the previous step
    loss.backward()                         # compute new gradients for this step
    optimizer.step()                        # apply the AdamW update rule described above

print(loss.item())


# %%
idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx=idx, max_new_tokens=500)[0].tolist()))


# %%
torch.manual_seed(42)

# A 3x3 lower-triangular matrix of 1s, then row-normalized so each row sums to 1.
# torch.tril zeroes out everything ABOVE the diagonal, keeping the diagonal and below.
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)   # divide each row by its own sum -> rows are averages

b = torch.randint(0, 10, (3, 2)).float()   # some arbitrary data: 3 "tokens", 2 features each
c = a @ b   # matrix-multiply the weight matrix against the data

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
B, T, C = 4, 8, 2   # batch, time, channels — 4 independent sequences, 8 tokens each, 2 features
x = torch.randn(B, T, C)
x.shape

# %%

xbow = torch.zeros((B, T, C))   # "bag of words" running average, same shape as x
for b in range(B):              # for every sequence in the batch...
    for t in range(T):          
        xprev = x[b, :t+1]                # all tokens from the start up to position t (t,C)
        xbow[b, t] = torch.mean(xprev, 0)  # average them along the time dimension

# %%
wei = torch.tril(torch.ones(T, T))           # (T,T) lower-triangular matrix of 1s
wei = wei / wei.sum(1, keepdim=True)          # normalize each row to sum to 1 (-> row averages)


xbow2 = wei @ x

# Confirm this matches the slow, explicit version exactly (up to floating point tolerance).
torch.allclose(xbow, xbow2)

# %%
tril = torch.tril(torch.ones(T, T))   # the same lower-triangular mask as before

wei = torch.zeros((T, T))             # start from all-zero "affinities" between every pair

wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)   # softmax of a row of [0,0,...,0,-inf,-inf,...] is just a


xbow3 = wei @ x
torch.allclose(xbow, xbow3)


# %%
torch.manual_seed(1337)
B, T, C = 4, 8, 32   # batch, time, channels — note C is now 32, a richer per-token embedding
x = torch.randn(B, T, C)

# let's see a single Head perform self-attention
head_size = 16


key   = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)     # (B, T, 16) — every token's "what I contain" vector
q = query(x)   # (B, T, 16) — every token's "what I'm looking for" vector


wei = q @ k.transpose(-2, -1)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))   # block attending to future positions
wei = F.softmax(wei, dim=-1)                       # turn affinities into a distribution per row

v = value(x)     # (B, T, 16) — every token's "what I'll communicate" vector
out = wei @ v    # weighted aggregation of VALUES, weighted by the learned affinities

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
wei_scaled = (q_demo @ k_demo.transpose(-2, -1)) * head_size**-0.5   # <- the fix: * 1/sqrt(d)

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
      self.gamma = torch.ones(dim)    # learned per-feature scale, initialized to 1 (no-op)
      self.beta = torch.zeros(dim)    # learned per-feature shift, initialized to 0 (no-op)

  def __call__(self, x):
      
      xmean = x.mean(1, keepdim=True)              # mean across features, per example
      xvar = x.var(1, keepdim=True)                 # variance across features, per example
      xhat = (x - xmean) / torch.sqrt(xvar + self.eps)   # normalize to zero mean, unit variance
      self.out = self.gamma * xhat + self.beta            # learned rescale/shift
      return self.out

  def parameters(self):
      return [self.gamma, self.beta]

torch.manual_seed(1337)
module = LayerNorm1d(100)
x = torch.randn(32, 100)   # batch size 32 of 100-dimensional vectors
x = module(x)
x.shape

# %%
x[:,0].mean(), x[:,0].std()

# %%
x[0,:].mean(), x[0,:].std()

# %%
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# %%
class Head(nn.Module):
    ''' one head of self-attention '''

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # a fixed causal mask, registered as a buffer (not a learnable parameter) so it
        # travels with the module across devices automatically
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)     # (B,T,C) -> really (B,T,head_size); C here means head_size
        q = self.query(x)   # (B,T,C)
        # compute attention scores ("affinities") -- scaled dot-product, as derived in Part 7
        wei = q @ k.transpose(-2,-1) * C**-0.5   # (B, T, C) @ (B, C, T) -> (B, T, T)
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
       
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
#%%
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
        self.proj = nn.Linear(n_embd, n_embd)   # mixes information across heads after concat
        self.proj.GPT_SCALE_INIT = 1   # flagged for scaled residual init, see GPTLanguageModel._init_weights
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
        proj = nn.Linear(4 * n_embd, n_embd)   # project back down: 4*n_embd -> n_embd
        proj.GPT_SCALE_INIT = 1   # flagged for scaled residual init, see GPTLanguageModel._init_weights
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # expand: n_embd -> 4*n_embd
            nn.GELU(),                        # nonlinearity (GPT-2 uses GELU, not ReLU)
            proj,
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
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)   # the "communication" step
        self.ffwd = FeedFoward(n_embd)                      # the "computation" step
        self.ln1 = nn.LayerNorm(n_embd)   # pre-norm before attention
        self.ln2 = nn.LayerNorm(n_embd)   # pre-norm before the feed-forward network

    def forward(self, x):
        x = x + self.sa(self.ln1(x))      # residual connection around attention
        x = x + self.ffwd(self.ln2(x))    # residual connection around the feed-forward net
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
tok_emb_demo = torch.randn(2, T_demo, n_embd)   # pretend token embeddings, batch of 2
combined = tok_emb_demo + pos_emb_demo
print(combined.shape)

# %%
batch_size = 64
max_iters = 1000
eval_interval = 250
learning_rate = 3e-4
eval_iters = 50

# %%
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()   # switch dropout/batchnorm (if any) into evaluation behavior
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()   # switch back to training behavior before returning
    return out

# %%
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)             # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)  # project back up to vocabulary size

        self.token_embedding_table.weight = self.lm_head.weight   # weight tying, as in GPT-2
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # GPT-2-style init: N(0, 0.02) everywhere, with residual-stream projections
        # scaled down by 1/sqrt(2 * n_layer) so the residual stream's variance doesn't
        # grow with depth (each block adds two such projections: attn.proj and ffwd's last linear)
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'GPT_SCALE_INIT'):
                std *= (2 * n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)                              # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb     # (B,T,C) -- content + position, summed (Part 13)
        x = self.blocks(x)        # (B,T,C) -- n_layer rounds of attend-then-compute (Part 12)
        x = self.ln_f(x)          # (B,T,C) -- final normalization
        logits = self.lm_head(x)  # (B,T,vocab_size) -- next-character logits at every position

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)   # identical to the bigram model's loss

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens -- required now, see explanation above
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# %%
torch.manual_seed(1337)

model = GPTLanguageModel()
m = model.to(device)   # move all parameters to GPU if available, no-op otherwise

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

warmup_iters = max(1, max_iters // 50)   # short linear warmup, then cosine decay to 0

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    decay_ratio = min((it - warmup_iters) / max(1, max_iters - warmup_iters), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   # 1 -> 0
    return coeff * learning_rate

# %%
checkpoint_path = 'gpt_model_best.pt'

if os.path.exists(checkpoint_path):
    # a trained checkpoint is already sitting on disk -- load it instead of
    # retraining from scratch every time this script runs
    print(f"loading existing checkpoint from {checkpoint_path} (delete this file to retrain from scratch)")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    best_val_loss = float('inf')

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), checkpoint_path)

        # set this step's learning rate (warmup + cosine decay)
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # avoid rare exploding-gradient spikes
        optimizer.step()

    model.load_state_dict(torch.load(checkpoint_path))   # reload best checkpoint, not just the last step

# %%
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


# %%
# ---- general-purpose runner: encode any prompt, generate, decode -----------------
def generate_text(prompt="", max_new_tokens=500):
    ''' feed the trained model a text prompt (or none) and return its completion '''
    model.eval()
    ids = [stoi[c] for c in prompt if c in stoi]   # silently drop characters unseen in training
    if not ids:
        ids = [0]   # empty/unrecognized prompt -> start from a single seed token, same as above
    context = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = m.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    model.train()
    return decode(out_ids)


# %%
if __name__ == "__main__":
    print("\nModel ready. Type a prompt and press Enter to generate text (blank line or 'quit' to exit).\n")
    while True:
        try:
            prompt = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip().lower() in ("", "quit", "exit"):
            break
        print(generate_text(prompt, max_new_tokens=300))
        print()
# %%
