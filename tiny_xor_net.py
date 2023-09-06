import tinytorch as tt


class XorNet(tt.Module):
    def __init__(self):
        super().__init__()
        self.l1 = tt.Linear(2, 2)
        self.l2 = tt.Linear(2, 1)

    def forward(self, x):
        x = self.l1(x)
        x = tt.tanh(x)
        x = self.l2(x)
        x = tt.tanh(x)
        return x


lr = 1e-1

model = XorNet()
optimizer = tt.SGD(model.parameters(), lr=lr)

x = tt.tensor(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ]
)
y = tt.tensor(
    [
        [0],
        [1],
        [1],
        [0],
    ]
)


ITER = 2000

pred = model(x)
print(pred)
for idx in range(ITER):
    pred = model(x)
    loss = tt.mse_loss(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss.item())

print("=" * 100)
pred = model(x)
print(pred)


class Max(Function):
    def forward(x: Tensor, shape):
        axis = tuple(idx for idx, (a, b) in enumerate(zip(x.shape, shape)) if a != b)
        return x.max(axis, keepdims=True)

    def backward(ctx: Function, grad_output: Tensor) -> Tensor:
        x, shape = ctx.args
        ret = Max.forward(x, shape)

        max_is_1s: np.ndarray = np.full_like(x.data, 1.0) - (
            x.data < np.broadcast_to(ret.data, x.shape)
        )
        axis = tuple(
            idx
            for idx, (a, b) in enumerate(zip(max_is_1s.shape, grad_output.shape))
            if a != b
        )
        div = max_is_1s.sum(axis, keepdims=True)
        div = np.broadcast_to(div, x.shape)

        maximum = max_is_1s / div
        maximum = maximum * np.broadcast_to(grad_output.data, x.shape)  #
        return maximum
