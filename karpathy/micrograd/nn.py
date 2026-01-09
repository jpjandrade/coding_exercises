import random

from engine import Value
from sample_data import sample_training, sample_eval


class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        logits = sum((w_i * x_i for w_i, x_i in zip(self.w, x)), self.b)
        out = logits.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in, n_outs):
        sizes = [n_in] + n_outs
        self.layers = [Layer(n_1, n_2) for n_1, n_2 in zip(sizes[:-1], sizes[1:])]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self, training_data, test_data, epochs, batch_size, learning_rate):
        eval_loss = self.evaluate(test_data)
        print(f"Pretrain: eval_loss {eval_loss}")
        for epoch in range(epochs):
            for i in range(0, len(training_data), batch_size):
                batch_data = training_data[i : i + batch_size]
                train_loss = self.sgd(batch_data, learning_rate)

            eval_loss = self.evaluate(test_data)
            print(f"Epoch {epoch}: loss {train_loss}, eval_loss {eval_loss}")

    def sgd(self, batch_data, learning_rate):
        total_loss = 0
        for x, y in batch_data:
            y_pred = self(x)

            loss: Value = (y - y_pred) ** 2
            loss.backward()

            total_loss += loss.data

        self.update_weights(learning_rate)
        self.zero_grad()

        return total_loss / len(batch_data)

    def update_weights(self, learning_rate):
        for p in self.parameters():
            p.data -= p.grad * learning_rate

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def evaluate(self, data):
        loss: Value = Value(0)
        for x, y in data:
            y_pred = self(x)
            loss += (y - y_pred) ** 2

        return loss.data / len(data)


net = MLP(3, [4, 4, 1])
net.train(sample_training, sample_eval, 10, 2, 0.1)
