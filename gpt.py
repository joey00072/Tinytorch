import tinytorch as tt
from dataclasses import dataclass
from typing import Any
import numpy as np
import random

# np.random.seed(0)
# curl -LO https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

@dataclass
class ModelArgs:
    seq_len: int = 10
    d_model: int = 16
    n_heads: int = 2
    vocab_size: int = 10
    num_layers: int = 2
    esp: int =1e-5


def silu(x) -> tt.Tensor:
    return x * tt.sigmoid(x)


class RMSNorm(tt.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = tt.Parameter(tt.ones((dim)))

    def _norm(self, x:tt.Tensor):
        rms = ((x**2).mean(axis=-1, keepdim=True) + self.eps) ** 0.5
        return x /rms

    def forward(self, x):
        import time
        # return x
        start:int  = time.perf_counter()
        output = self._norm(x)
        # output = x
        return output * self.weight


class Embedding(tt.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # print(f"{self.num_embeddings=} {self.embedding_dim=}")

        self.weight = tt.Parameter(tt.rand((num_embeddings, embedding_dim)))

    def forward(self, x: tt.Tensor):
        # print(self.weight.data.sum())

        shape = x.shape
        x = x.reshape(-1, 1)
        tensors = []
        for idx, scaler in enumerate(x):
            # print(f"{self.num_embeddings} {scaler.item()=}")
            tensor = self.weight[scaler.item()]
            tensors.append(tensor)
        tensor = tt.stack(tensors)
        return tensor.reshape(*shape, self.embedding_dim)


class MLP(tt.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, expansion: int = 4
    ) -> None:
        super().__init__()
        self.w1 = tt.Linear(in_features, in_features * expansion)
        self.w2 = tt.Linear(in_features, in_features * expansion)
        self.w3 = tt.Linear(in_features * expansion, out_features)

    def forward(self, x):
        return self.w3(silu(self.w1(x)) * self.w2(x))


class MHA(tt.Module):
    def __init__(self, model_args: ModelArgs) -> None:
        super().__init__()
        self.key = tt.Linear(model_args.d_model, model_args.d_model)
        self.query = tt.Linear(model_args.d_model, model_args.d_model)
        self.value = tt.Linear(model_args.d_model, model_args.d_model)
        self.proj = tt.Linear(model_args.d_model, model_args.d_model)

        self.n_heads = model_args.n_heads

    def forward(self, x: tt.Tensor, mask=None):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k: tt.Tensor = k.reshape(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.reshape(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        wei = q @ k.transpose(-1, -2)
        if mask is not None:
            wei = mask[ :, :T, :T] * wei
        v = wei @ v
        v = v.reshape(B, T, C)
        x = self.proj(v)
        return x


class Block(tt.Module):
    def __init__(self, model_args: ModelArgs) -> None:
        super().__init__()
        self.attn = MHA(model_args)
        self.ffn = MLP(model_args.d_model, model_args.d_model)
        self.l1 = RMSNorm(model_args.d_model, eps=model_args.esp)
        self.l2 = RMSNorm(model_args.d_model, eps=model_args.esp)
        self.mask = tt.tensor(
            np.tril(np.ones((model_args.n_heads, model_args.seq_len, model_args.seq_len)))
        )

    def forward(self, x):
        x = x + self.attn(self.l1(x), self.mask)
        x = x + self.ffn(self.l1(x))
        return x


class GPT(tt.Module):
    def __init__(self, model_args: ModelArgs) -> None:
        self.tok_embeddings = Embedding(model_args.vocab_size, model_args.d_model)
        self.pos_embeddings = Embedding(model_args.seq_len, model_args.d_model)

        self.blocks = tt.ModuleList(
            [Block(model_args) for _ in range(model_args.num_layers)]
        )

        self.norm = RMSNorm(model_args.d_model,model_args.esp)
        self.proj = tt.Linear(model_args.d_model,model_args.vocab_size)

    def forward(self, x: tt.Tensor):
        tokens = self.tok_embeddings(x)
        pos = self.pos_embeddings(tt.arange(x.shape[1]))
        x = tokens + pos
        # print(f"{x=}")

        for block in self.blocks:
            x = block(x)
            # print(f"{x=}")

        x = self.norm(x)
        # print(f"{x=}")
        logits = self.proj(x)
        return logits

def model_size(model):
    num_params = sum([p.size for p in model.parameters()])
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)

class Tokenizer:
    def __init__(self, text_file):
        with open(text_file, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
    
    def train_val_split(self, ratio=0.9):
        data = self.encode(self.text)
        n = int(ratio * len(data))
        train_data = data[:n]
        val_data = data[n:]
        print(f"{len(data)=}")
        return train_data, val_data



class TextDataset:
    def __init__(self, data, batch_size, seq_len):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.data_len = len(self.data)
        print(self.data_len)
    
    def __len__(self):
        return (self.data_len - self.seq_len) // self.batch_size

    def __iter__(self):
        # Shuffle indices for random sampling
        indices = list(range(0, self.data_len - self.seq_len))
        random.shuffle(indices)

        for i in range(0, len(indices)):
            batch_indices = indices[i:i+self.batch_size]
            
            x_batch = [self.data[idx:idx+self.seq_len] for idx in batch_indices]
            y_batch = [self.data[idx+1:idx+self.seq_len+1] for idx in batch_indices]
            
            yield tt.tensor(x_batch), tt.tensor(y_batch)


def generate(tokenizer:Tokenizer,model:GPT,x=None,ouput_len=100):
    
    if x is None:
        x = tt.tensor([[[1]]])
    for _ in range(ouput_len):
        char = tokenizer.decode([x.data[0][-1].sum()])
        # print(char, end='', flush=True)
        # print(f"{x.shape=}")
        pred = model.forward(x.reshape(-1,1))
        o = tt.tensor([[[np.argmax(pred[0][-1].data)]]])
        # print(f"I: {x.shape=} {o.shape=}")
        s = [[i.sum() for i in x.data[0]]+[i.sum() for i in o.data[0]]]
        print(tokenizer.decode(s[0])[-1],end='', flush=True)
        x = tt.tensor(s)
    
    

def train(model, optimizer:tt.Optimizer, dataset: TextDataset, num_iterations):
    iteration = 0
    for data,target in dataset:

        B, T = data.shape

        # Forward pass
        logits:tt.Tensor = model(data)
        
        # print(f"{data.shape=}")
        
        # Reshape logits and targets for loss computation
        logits = logits.reshape(B * T, -1)
        target = target.reshape(B * T)
        

        # Compute loss
        loss = tt.cross_entropy(logits, target)

        # print(f"{logits.shape=} {target.shape=} {loss.shape=}")

        # Zero gradients, backward pass, optimizer step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Iteration [{iteration + 1}/{num_iterations}], Loss: {loss.item()}")

        if iteration%10==0:
            generate(tokenizer,model)
        iteration += 1
        if iteration >= num_iterations:
            print("DONE...")
            break

    print("Training complete.")




# Define some hyperparameters
learning_rate = 3e-3
batch_size = 8
seq_len = 10
d_model = 32
n_heads = 2
num_layers = 2
# Example Usage
tokenizer = Tokenizer('input.txt')
train_data, val_data = tokenizer.train_val_split()

batch_size = 8
seq_len = 10

train_dataset = TextDataset(train_data, batch_size, seq_len)
val_dataset = TextDataset(val_data, batch_size, seq_len)

model_args = ModelArgs(
    d_model=d_model, seq_len=seq_len, vocab_size=tokenizer.vocab_size, n_heads=n_heads
)
# Initialize model and optimizer
model = GPT(model_args)
# print(model.state_dict())
print(f"Model: { model_size(model)}")
optimizer = tt.Adam(model.parameters(), lr=learning_rate)

# Call the train function to initiate training
train(model, optimizer, train_dataset, num_iterations=100_000)  # Replace 1000 with the actual number of iterations you want

generate(tokenizer,model)

# Tokenizer Class







