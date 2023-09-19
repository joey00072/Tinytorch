import tinytorch as torch
import tinytorch as nn
import tinytorch as optim
import tinytorch as F

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

from dataclasses import dataclass
import numpy as np
import math
import numpy

# hyperparameters
batch_size = 64
block_size = 128
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 50
n_embd = 128*2
n_head = 4
n_layer = 2
dropout = 0.2
# ------------


# torch.manual_seed(1337)


class CharTokenizer:
    def __init__(self, text=None, filepath=None):
        self.text = text
        if filepath:
            with open(filepath, "r", encoding="utf-8") as f:
                self.text = f.read()
        elif text is None:
            raise ValueError("Either text or filepath must be provided.")

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return "".join([self.itos[i] for i in l])


tokenizer = CharTokenizer(filepath="input.txt")

# Train and test splits
data = torch.tensor(tokenizer.encode(tokenizer.text)).long()
n = int(0.95 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    len_data = len(data)
    ix = np.random.randint(0, len_data - block_size, batch_size)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@dataclass
class ModelArgs:
    seq_len: int = 10
    d_model: int = 16
    n_heads: int = 2
    vocab_size: int = 10
    num_layers: int = 2
    esp: float = 1e-5


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.rand((num_embeddings, embedding_dim)) / embedding_dim
        )

    def forward(self, x: torch.Tensor):
        return self.weight[x]


def silu(x) -> torch.Tensor:
    return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((dim)))

    def _norm(self, x: torch.Tensor):
        rms = ((x**2).mean(axis=-1, keepdim=True) + self.eps) ** 0.5
        return x / rms

    def forward(self, x):
        output = self._norm(x)
        return output * self.weight


# @torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    
    for split in ["train", "val"]:
        losses = []
        for k in range(eval_iters):
            
            data, targets = get_batch(split)
            logits = model(data)

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            losses.append(loss.item())
        out[split] = sum(losses)/len(losses)
    model.train()
    return out


class MHA(nn.Module):
    def __init__(self, model_args: ModelArgs) -> None:
        super().__init__()
        self.key = nn.Linear(model_args.d_model, model_args.d_model)
        self.query = nn.Linear(model_args.d_model, model_args.d_model)
        self.value = nn.Linear(model_args.d_model, model_args.d_model)
        self.proj = nn.Linear(model_args.d_model, model_args.d_model)
        self.head_dim = model_args.d_model // model_args.n_heads

        self.n_heads = model_args.n_heads
        mask = torch.tensor(
            (
                np.tril(np.zeros((1, 1, model_args.seq_len, model_args.seq_len)))
                + np.triu(
                    -np.inf * np.ones((1, 1, model_args.seq_len, model_args.seq_len)),
                    k=1,
                )
            )
        ).float()

        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.reshape(B, T, self.n_heads, C // self.n_heads)
        q = q.reshape(B, T, self.n_heads, C // self.n_heads)
        v = v.reshape(B, T, self.n_heads, C // self.n_heads)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        wei = (q @ k.transpose(-1, -2)) / (self.head_dim**0.5)
        wei = self.mask[:, :, :T, :T] + wei
        wei = F.softmax(wei, dim=-1)

        v = wei @ v
        v = v.transpose(1, 2).reshape(B, T, C)

        x = self.proj(v)
        return x


class MLP(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, expansion: int = 4
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_features, in_features * expansion)
        self.w2 = nn.Linear(in_features, in_features * expansion)
        self.w3 = nn.Linear(in_features * expansion, out_features)

    def forward(self, x):
        return self.w3(silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, model_args: ModelArgs) -> None:
        super().__init__()
        self.attn = MHA(model_args)
        self.ffn = MLP(model_args.d_model, model_args.d_model)
        self.l1 = RMSNorm(model_args.d_model, eps=model_args.esp)
        self.l2 = RMSNorm(model_args.d_model, eps=model_args.esp)

    def forward(self, x):
        x = x + self.attn(self.l1(x))
        x = x + self.ffn(self.l2(x))
        return x


class GPT(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.token_embedding = Embedding(model_args.vocab_size, model_args.d_model)
        self.position_embedding = Embedding(model_args.seq_len, model_args.d_model)

        self.layers = nn.ModuleList(
            [Block(model_args) for _ in range(model_args.num_layers)]
        )
        self.norm = RMSNorm(model_args.d_model)
        self.proj = nn.Linear(model_args.d_model, model_args.vocab_size)

    def forward(self, x):
        B, T = x.shape

        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T).to(device))
        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.proj(x)

        return logits

    def generate(self, idx, max_new_tokens):
        self.eval()
        for i in range(max_new_tokens):
            # print(i)
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx


model_args = ModelArgs(
    d_model=n_embd,
    seq_len=block_size,
    vocab_size=tokenizer.vocab_size,
    n_heads=n_head,
    num_layers=n_layer,
)

model = GPT(model_args)
m = model.to(device)

print(sum(math.prod(p.shape) for p in m.parameters()) / 1e6, "M parameters")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for iter in range(1,max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        print("###")
        model.eval()
        context = torch.zeros((1, 1)).to(device).long()
        print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
        model.train()
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train'].item():.4f}, val loss {losses['val'].item():.4f}"
        )
        optimizer.zero_grad()

    data, targets = get_batch("train")
    logits = model(data)

    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = targets.view(B * T)

    loss = F.cross_entropy(logits, targets)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    

    print(f"{iter=}")

context = torch.zeros((1, 1)).to(device).long()
print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
