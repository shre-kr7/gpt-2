##THIS WHOLE CODE IS THE COPY OF ANDREJ KARPATHY'S GPT TUTORIAL.
## IT IS FOR REFERNCE PURPOSES ONLY.

#%%
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# %%
print("length of dataset in characters: ", len(text))


# %%
# `set(text)` collects every distinct character that appears anywhere in the dataset.
# sorted(...) gives us a fixed, reproducible ordering (important: we need the SAME mapping
# every time we run this, otherwise saved models would become meaningless).
chars = sorted(list(set(text)))

# The vocabulary size is just how many distinct characters exist in our dataset.
vocab_size = len(chars)

print(''.join(chars))  # all the characters, concatenated, so we can eyeball them
print(vocab_size)


# %%
# stoi = "string to integer": a dict mapping each character to its index in `chars`.
# itos = "integer to string": the reverse mapping, index back to character.
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# encode: take a Python string, return a list of integers (one per character).
encode = lambda s: [stoi[c] for c in s]

# decode: take a list of integers, return the string they represent.
decode = lambda l: ''.join([itos[i] for i in l])

# Round-trip sanity check: encoding then decoding should give back the original string.
print(encode("hii there"))
print(decode(encode("hii there")))


# %%
import torch  # we use PyTorch: https://pytorch.org

# Encode the ENTIRE dataset (all 1.1M characters) into a single 1-D tensor of integers.
# dtype=torch.long because these are integer IDs going into an embedding lookup later
# (PyTorch embeddings require integer index tensors, conventionally int64 / "long").
data = torch.tensor(encode(text), dtype=torch.long)

print(data.shape, data.dtype)
print(data[:1000])  # the same first 1000 characters we printed above, now as integer IDs


# %%
# Split the data into a training set and a validation set.
# We hold out the LAST 10% of the text as validation, training on the first 90%.
# Why hold out anything at all? A model can always drive train loss to ~0 by memorizing the
# training text outright. Held-out validation loss is what tells us whether the model has
# learned generalizable patterns (e.g. "q is usually followed by u", "this looks like a
# character name and then a colon") rather than just memorizing exact passages.
n = int(0.9*len(data))       # index marking the 90% cutoff point
train_data = data[:n]        # first 90% of the tokens
val_data = data[n:]          # last 10% of the tokens


# %%
block_size = 8
# Grab the first block_size+1 tokens. We need block_size+1, not block_size, because this one
# chunk of 9 characters actually encodes 8 separate (input, target) pairs (see below).
train_data[:block_size+1]

# %%
# A block of length block_size+1 gives us:
#   x = the first block_size tokens   -> what the model sees
#   y = the tokens shifted by one     -> what the model should predict at each position
x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]   # everything up to and including position t
    target = y[t]        # the character that actually comes next
    print(f"when input is {context} the target: {target}")


# %%
torch.manual_seed(1337)   # fixes all of torch's randomness below, so results are reproducible
batch_size = 4   # how many independent sequences will we process in parallel?
block_size = 8   # what is the maximum context length for predictions?

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

# Same idea as the single-example loop above, but now over every example in the batch too.
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
        # nn.Embedding(vocab_size, vocab_size) is a (vocab_size, vocab_size) matrix of
        # learnable parameters. Given an integer token id, it returns that row of the matrix.
        # Here we (ab)use it as a direct "current token -> next token logits" lookup table:
        # each token directly reads off the logits for the next token from a lookup table.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensors of integers (B=batch, T=time/sequence length)

        # Look up a row of the table for every single token in idx. Each row is a length
        # vocab_size vector of raw, unnormalized scores ("logits") for what comes next.
        # Shape: (B, T) -> (B, T, vocab_size). We name the last dimension C ("channels").
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            # At generation time we don't have a "correct answer" to compare against,
            # so there's nothing to compute a loss against.
            loss = None
        else:
            B, T, C = logits.shape
            # F.cross_entropy expects a 2D input of shape (N, C) and a 1D target of shape (N,),
            # so we flatten the batch and time dimensions together into one big batch of
            # individual (token, next_token) prediction problems.
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Cross-entropy = -log(probability assigned to the correct next token), averaged
            # over all B*T predictions. This is THE loss function for next-token prediction:
            # internally it applies softmax to `logits` and then takes -log of the probability
            # at the `targets` index. Lower is better; 0 would mean perfect, fully-confident
            # predictions every time.
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
# Start the generation from a single token: index 0, which decodes to '\n' (the first
# character in our sorted vocabulary). Shape (1, 1): batch size 1, sequence length 1.
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
# Toy example illustrating how matrix multiplication can be used for a "weighted aggregation".
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
# Consider the following toy example:
torch.manual_seed(1337)
B, T, C = 4, 8, 2   # batch, time, channels — 4 independent sequences, 8 tokens each, 2 features
x = torch.randn(B, T, C)
x.shape

# %%
# We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B, T, C))   # "bag of words" running average, same shape as x
for b in range(B):              # for every sequence in the batch...
    for t in range(T):          # ...and every position in that sequence...
        xprev = x[b, :t+1]                # all tokens from the start up to position t (t,C)
        xbow[b, t] = torch.mean(xprev, 0)  # average them along the time dimension

# %%
# version 2: using matrix multiply for a weighted aggregation
wei = torch.tril(torch.ones(T, T))           # (T,T) lower-triangular matrix of 1s
wei = wei / wei.sum(1, keepdim=True)          # normalize each row to sum to 1 (-> row averages)

# (T,T) @ (B,T,C): PyTorch broadcasts `wei` across the batch dimension, effectively doing
# (B, T, T) @ (B, T, C) -> (B, T, C) — the same averaging, applied independently to every
# sequence in the batch, with no Python-level loop at all.
xbow2 = wei @ x

# Confirm this matches the slow, explicit version exactly (up to floating point tolerance).
torch.allclose(xbow, xbow2)

# %%
# version 3: use Softmax
tril = torch.tril(torch.ones(T, T))   # the same lower-triangular mask as before

wei = torch.zeros((T, T))             # start from all-zero "affinities" between every pair
# masked_fill replaces every entry where tril==0 (i.e. the upper triangle, "the future")
# with -inf. After softmax, -inf becomes exactly 0 probability — those positions are
# completely excluded.
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)   # softmax of a row of [0,0,...,0,-inf,-inf,...] is just a
                                 # uniform distribution over the non--inf entries

xbow3 = wei @ x
torch.allclose(xbow, xbow3)

# %%
torch.manual_seed(1337)
B, T, C = 4, 8, 32   # batch, time, channels — note C is now 32, a richer per-token embedding
x = torch.randn(B, T, C)

# let's see a single Head perform self-attention
head_size = 16

# Three independent linear layers, each projecting from C=32 down to head_size=16.
# bias=False: standard for these projections in Transformers — we don't need a learnable
# additive offset here, just a learned linear transformation of the embedding.
key   = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)     # (B, T, 16) — every token's "what I contain" vector
q = query(x)   # (B, T, 16) — every token's "what I'm looking for" vector

# Dot-product affinity between every pair of positions: for each batch, a (T,T) matrix where
# entry [i,j] = q[i] . k[j], i.e. "how much does token i's query match token j's key."
# (B, T, 16) @ (B, 16, T) ---> (B, T, T)  [transpose swaps the last two dims of k]
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
wei_demo = q_demo @ k_demo.transpose(-2, -1)   # NOTE: no scaling applied yet

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
# Re-run the real Head computation from earlier, this time with scaling included.
wei = q @ k.transpose(-2, -1) * head_size**-0.5    # <- the only change from before
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

out = wei @ v
out.shape

# %%
class LayerNorm1d:  # a from-scratch implementation, to make the mechanics fully explicit
                     # (used to be called BatchNorm1d in earlier course material — note how
                     # little code actually changes between the two; only WHICH dimension
                     # we reduce over)

  def __init__(self, dim, eps=1e-5, momentum=0.1):
      self.eps = eps
      self.gamma = torch.ones(dim)    # learned per-feature scale, initialized to 1 (no-op)
      self.beta = torch.zeros(dim)    # learned per-feature shift, initialized to 0 (no-op)

  def __call__(self, x):
      # calculate the forward pass
      # dim=1 is the FEATURE dimension here (x has shape (batch, features)) — this is the
      # key difference from BatchNorm, which would reduce over dim=0 (the batch dimension)
      # instead.
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
block_size = 32   # what is the maximum context length for predictions?
n_embd = 64       # embedding dimension for every token
n_head = 4         # number of attention heads per block
n_layer = 4        # number of Transformer blocks stacked on top of each other
dropout = 0.0      # dropout probability

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)

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
        # causal mask: only attend to this position and earlier ones.
        # self.tril[:T, :T] handles sequences shorter than block_size (e.g. during generation,
        # before the context has grown to the full block_size).
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

# %%
# Smoke test: does a single Head run correctly on dummy data of the right shape, and does it
# produce the shape we expect? We use a 2-example "batch" of 10-token sequences, each token
# represented by an n_embd-dimensional vector -- exactly the shape that will flow out of the
# token+position embedding layers later.
head_size = n_embd // n_head
dummy_x = torch.randn(2, 10, n_embd)
test_head = Head(head_size)
print(test_head(dummy_x).shape)   # expect (2, 10, head_size) = (2, 10, 16)

# %%
class MultiHeadAttention(nn.Module):
    ''' multiple heads of self-attention in parallel '''

    def __init__(self, num_heads, head_size):
        super().__init__()
        # nn.ModuleList (not a plain Python list!) so PyTorch correctly tracks every Head's
        # parameters as part of this module (needed for .parameters(), .to(device), etc.)
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)   # mixes information across heads after concat
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # run every head on the same input, then concatenate their outputs along the last
        # (feature) dimension: num_heads tensors of shape (B,T,head_size) -> (B,T,n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# %%
# Smoke test: n_head heads of head_size each should recombine into n_embd output features.
test_mha = MultiHeadAttention(n_head, head_size)
print(test_mha(dummy_x).shape)   # expect (2, 10, n_embd) = (2, 10, 64)


# %%
class FeedFoward(nn.Module):
    ''' a simple linear layer followed by a non-linearity '''

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # expand: n_embd -> 4*n_embd
            nn.ReLU(),                        # nonlinearity
            nn.Linear(4 * n_embd, n_embd),    # project back down: 4*n_embd -> n_embd
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# %%
# Smoke test: shape should be unchanged end-to-end (n_embd in, n_embd out), since this module
# slots into the residual stream alongside attention and must match its shape exactly.
test_ffwd = FeedFoward(n_embd)
print(test_ffwd(dummy_x).shape)   # expect (2, 10, n_embd) = (2, 10, 64)


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
# Smoke test: stack a couple of Blocks (just like nn.Sequential will do for real, shortly)
# and confirm the shape is preserved exactly -- this is required for residual connections to
# even be well-defined (x and f(x) must have the same shape to be added together).
test_blocks = nn.Sequential(Block(n_embd, n_head), Block(n_embd, n_head))
print(test_blocks(dummy_x).shape)   # expect (2, 10, n_embd) = (2, 10, 64), unchanged


# %%
# A learned table mapping "position index" -> a learned n_embd-dimensional vector.
# Note this table has block_size rows, NOT vocab_size rows -- it's indexed by WHERE a token
# is in the sequence (0, 1, 2, ..., block_size-1), not by WHICH token it is.
position_embedding_table = nn.Embedding(block_size, n_embd)

T_demo = 10
# torch.arange(T_demo) = [0, 1, 2, ..., 9] -- the position indices for a 10-token sequence
pos_emb_demo = position_embedding_table(torch.arange(T_demo))
print(pos_emb_demo.shape)   # (T, n_embd) = (10, 64) -- ONE vector per position, no batch dim


# %%
# token embeddings have shape (B, T, n_embd); position embeddings have shape (T, n_embd).
# Adding them relies on broadcasting: PyTorch automatically repeats the (T, n_embd) position
# tensor across the batch dimension, so every sequence in the batch gets the same positional
# signal added in, while each token's own content embedding stays unique to that token.
tok_emb_demo = torch.randn(2, T_demo, n_embd)   # pretend token embeddings, batch of 2
combined = tok_emb_demo + pos_emb_demo
print(combined.shape)   # (2, 10, 64) -- shape unchanged, position info now mixed in

#%%
batch_size = 16        # how many independent sequences will we process in parallel?
max_iters = 5000        # total number of training steps
eval_interval = 500      # how often (in steps) to report train/val loss
learning_rate = 1e-3      # AdamW's step size (see Part 5)
eval_iters = 200   
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
# create a PyTorch optimizer -- AdamW, exactly as derived and used in Part 5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# %%
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

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
