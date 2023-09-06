# Standard library imports
import os
import random
import struct
import time
import gzip

# Third-party imports
import numpy as np
import requests
from tqdm import tqdm
import tinytorch as tt

# Constants
EPOCHS = 3
BATCH_SIZE = 32
LR = 4e-3
MNIST_DIR = "mnist"
BASE_URL = "http://yann.lecun.com/exdb/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download_mnist():
    if not os.path.exists(MNIST_DIR):
        os.makedirs(MNIST_DIR)
        for file in FILES:
            url = f"{BASE_URL}{file}"
            response = requests.get(url)
            with open(f"{MNIST_DIR}/{file}", "wb") as f:
                f.write(response.content)


def load_mnist() -> tuple:
    def read_labels(filename: str) -> np.array:
        with gzip.open(filename, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    def read_images(filename: str) -> np.array:
        with gzip.open(filename, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            return images.reshape(num, rows, cols, 1)

    train_labels = read_labels(f"{MNIST_DIR}/train-labels-idx1-ubyte.gz")
    train_images = read_images(f"{MNIST_DIR}/train-images-idx3-ubyte.gz")
    test_labels = read_labels(f"{MNIST_DIR}/t10k-labels-idx1-ubyte.gz")
    test_images = read_images(f"{MNIST_DIR}/t10k-images-idx3-ubyte.gz")

    return (train_images, train_labels), (test_images, test_labels)


def one_hot(labels: np.array) -> np.array:
    return np.eye(10)[labels]


def get_batch(images: tt.Tensor, labels: tt.Tensor):
    indices = list(range(0, len(images), BATCH_SIZE))
    random.shuffle(indices)
    for i in indices:
        yield images[i : i + BATCH_SIZE], labels[i : i + BATCH_SIZE]


class Network(tt.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = tt.Linear(28 * 28, 128)
        self.l2 = tt.Linear(128, 10)

    def forward(self, x: tt.Tensor) -> tt.Tensor:
        x = tt.tanh(self.l1(x))
        return tt.tanh(self.l2(x))


def test(model: Network, test_images: tt.Tensor, test_labels: tt.Tensor):
    num_correct = 0
    for image, label in zip(test_images, test_labels):
        pred = model.forward(image)
        if np.argmax(pred.data) == np.argmax(label.data):
            num_correct += 1
    print(f"Test accuracy: {num_correct / len(test_images):.2%}")


def train(
    model: Network, optimizer: tt.Adam, train_images: tt.Tensor, train_labels: tt.Tensor
):
    for epoch in range(EPOCHS):
        # Create a tqdm object for the progress bar
        batch_generator = get_batch(train_images, train_labels)
        num_batches = len(train_images) // BATCH_SIZE
        with tqdm(total=num_batches) as pbar:
            for batch_images, batch_labels in batch_generator:
                optimizer.zero_grad()
                pred = model.forward(batch_images)
                loss = tt.mse_loss(pred, batch_labels)
                loss.backward()
                optimizer.step()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({"loss": float(loss.item())})

        print(f"Epoch: {epoch}, Loss: {loss}")
        test(model, test_images, test_labels)


if __name__ == "__main__":
    download_mnist()
    (train_images, train_labels), (test_images, test_labels) = load_mnist()

    train_labels, test_labels = map(
        tt.tensor, map(one_hot, [train_labels, test_labels])
    )
    train_images = tt.tensor(train_images.reshape(-1, 28 * 28) / 255)
    test_images = tt.tensor(test_images.reshape(-1, 28 * 28) / 255)

    model = Network()
    optimizer = tt.Adam(model.parameters(), lr=LR)

    start_time = time.perf_counter()
    train(model, optimizer, train_images, train_labels)
    print(f"Time to train: {time.perf_counter() - start_time} seconds")
