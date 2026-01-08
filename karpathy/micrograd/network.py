"""
Adapted from http://neuralnetworksanddeeplearning.com.
In this repo as a comparison of a backprop done by hand
vs an autograd, for my own learning :).
"""

import random
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid function."""
    sig = sigmoid(z)
    return sig * (1 - sig)


class Network:

    def __init__(self, sizes: list[int]):
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """Return the output of the network for input a."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def sgd(self, training_data, epochs, batch_size, lr, test_data=None):
        n = len(training_data)
        if test_data:
            print(f"Before start: {self.evaluate(test_data)} / {len(test_data)}")
        for epoch in range(epochs):
            random.shuffle(training_data)
            batches = [
                training_data[k : k + batch_size] for k in range(0, n, batch_size)
            ]

            for batch in batches:
                self.update_batch(batch, lr)

            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {len(test_data)}")
            else:
                print(f"Epoch {epoch} complete!")

    def update_batch(self, batch, lr):
        total_gradient_b = [np.zeros(b.shape) for b in self.biases]
        total_gradient_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            gradient_b, gradient_w = self.backprop(x, y)
            for i in range(len(total_gradient_b)):
                total_gradient_b[i] += gradient_b[i]
                total_gradient_w[i] += gradient_w[i]

        learning_rate = lr / len(batch)
        for i in range(len(self.biases)):
            self.biases[i] -= learning_rate * total_gradient_b[i]
            self.weights[i] -= learning_rate * total_gradient_w[i]

    def backprop(self, x, y):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations: list[np.ndarray] = [x]
        zs: list[np.ndarray] = []

        # Forward pass
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])

        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self.num_layers):
            delta = np.dot(
                self.weights[-layer + 1].transpose(), delta
            ) * sigmoid_derivative(zs[-layer])
            gradient_b[-layer] = delta
            gradient_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())

        return (gradient_b, gradient_w)

    def cost_derivative(self, a, y):
        return a - y

    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.feedforward(x)), np.argmax(y.flatten()))
            for (x, y) in test_data
        ]
        return np.sum([x == y for (x, y) in test_results])


def main():
    """
    Load MNIST dataset and train the network using SGD.
    Requires: pip install torch torchvision
    """
    from torchvision import datasets, transforms

    # Download and load MNIST
    print("Downloading MNIST...")
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    print("Done!")

    # Convert PyTorch dataset to format expected by Network.SGD
    # Network expects list of tuples (x, y) where:
    # - x is (784, 1) numpy array (flattened image)
    # - y is (10, 1) numpy array (one-hot encoded label)
    def prepare_data(dataset):
        data = []
        for img, label in dataset:
            # Flatten image from (1, 28, 28) to (784, 1)
            x = img.numpy().reshape(784, 1)
            # One-hot encode label
            y = np.zeros((10, 1))
            y[label] = 1.0
            data.append((x, y))
        
        return data

    print("Preparing training data...")
    training_data = prepare_data(train_dataset)
    print("Preparing test data...")
    test_data = prepare_data(test_dataset)

    # Create network: 784 inputs, 30 hidden neurons, 10 outputs
    net = Network([784, 128, 30, 10])

    # Train with SGD
    print("Starting training...")
    net.sgd(training_data, epochs=10, batch_size=32, lr=3.0, test_data=test_data)

    print("Training complete!")


if __name__ == "__main__":
    main()
