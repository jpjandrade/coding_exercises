import numpy as np


class MLP:
    def __init__(self, C):
        self.C = C
        self.w_in = np.random.randn(C, 4 * C) / np.sqrt(C)
        self.w_out = np.random.randn(4 * C, C) / np.sqrt(4 * C)
        self.b_in = np.zeros(4 * C)
        self.b_out = np.zeros(C)

        self.d_w_in = np.zeros_like(self.w_in)
        self.d_w_out = np.zeros_like(self.w_out)
        self.d_b_in = np.zeros_like(self.b_in)
        self.d_b_out = np.zeros_like(self.b_out)

        self.weights_with_grad = {
            "w_in": [self.w_in, self.d_w_in],
            "w_out": [self.w_out, self.d_w_out],
            "b_in": [self.b_in, self.d_b_in],
            "b_out": [self.b_out, self.d_b_out],
        }

    def forward(self, x: np.ndarray):
        # [B, T, C] @ [C, 4 * C] -> [B, T, 4 * C]
        z_in = np.matmul(x, self.w_in) + self.b_in

        # ReLU. Still [B, T, 4 * C]
        # Most transformers use more complicated ones
        # but I don't want to do the backward pass on these.
        a_in = np.maximum(z_in, 0)

        # Back to [B, T, C]
        z_out = np.matmul(a_in, self.w_out) + self.b_out

        self.cache: dict[str, np.ndarray] = {
            "x": x,
            "z_in": z_in,
            "a_in": a_in,
        }

        return z_out

    def backward(self, d_out: np.ndarray):
        if not self.cache:
            raise RuntimeError(
                "backward() called with no cache, run forward pass first."
            )

        x = self.cache["x"]
        a_in = self.cache["a_in"]
        z_in = self.cache["z_in"]

        # last op: z_out = a_in @ W_out.
        # We need d_W_out for the gradients and
        # d_a_in for the backward prop.

        # dL/d_wout =  d_out/d_wout dL/d_out = a_in^T @ dY
        # [4 * C, C] = [B, 4 * C, T] * [B, T, C] summed over 1st axis
        self.d_w_out += np.matmul(a_in.transpose(0, 2, 1), d_out).sum(axis=0)

        # For the linear term it's just [B, T, C] summed over B, T.
        self.d_b_out += d_out.sum(axis=(0, 1))  # sum over B and T

        # dL / da_in = \sum_k dL/d_out_k W_jk = dY @ W^T
        # [B, T, 4 * C] = [B, T, C] @ [C, 4 * C]
        d_a_in = np.matmul(d_out, self.w_out.T)

        # d_z_in is just backward prop of the relu:
        # dL / d_z_in = dL / d_a_in d_a_in/d_z_in = dL_d_a_in * I(z_in > 0)
        # [B, T, 4 * C] -> [B, T, 4 * C].
        d_z_in = d_a_in * (z_in > 0)

        # dL / d_win = d_L / dz_in d_z_in / d_x_in

        # [C, 4*C] = sum over B of ([B, C, T] @ [B, T, 4 * C])
        self.d_w_in += np.matmul(x.transpose(0, 2, 1), d_z_in).sum(axis=0)
        # Linear term, 4 * C = [B, T, 4 * C] summed over B, T
        self.d_b_in += d_z_in.sum(axis=(0, 1))

        # We need to return d_x, so we do:
        d_x = np.matmul(d_z_in, self.w_in.T)

        self.cache = {}

        return d_x

    def zero_grad(self):
        self.d_w_in.fill(0)
        self.d_w_out.fill(0)
        self.d_b_in.fill(0)
        self.d_b_out.fill(0)

    def step(self, lr):
        # TODO: implement momentum.
        self.w_in -= lr * self.d_w_in
        self.w_out -= lr * self.d_w_out
        self.b_in -= lr * self.d_b_in  # if using biases
        self.b_out -= lr * self.d_b_out
