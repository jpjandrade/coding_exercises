import numpy as np


class CausalSelfAttention:
    def __init__(self, T, C, n_head):
        self.w_qkv = np.random.randn(C, 3 * C) / np.sqrt(C)
        self.w_o = np.random.randn(C, C) / np.sqrt(C)

        self.d_w_qkv = np.zeros_like(self.w_qkv)
        self.d_w_o = np.zeros_like(self.w_o)

        self.n_head = n_head

        self.mask = np.triu(np.full((T, T), -np.inf), k=1)

        self.weights_with_grad = {
            "w_qkv": [self.w_qkv, self.d_w_qkv],
            "w_o": [self.w_o, self.d_w_o]
        }

    def forward(self, x: np.ndarray):
        B, T, C = x.shape
        n_dim = C // self.n_head

        # QKV is [B, T, 3 * C]
        qkv = np.matmul(x, self.w_qkv)

        # Each is [B, T, C]
        q, k, v = np.split(qkv, 3, axis=-1)

        # B, T, C -> B, n_head, T, n_dim
        q = q.reshape(B, T, self.n_head, n_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, n_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, n_dim).transpose(0, 2, 1, 3)

        # q is B, n_head, T, n_dim and k is B, n_head, n_dim, T
        # att_logits is B, n_head, T, T and we divide over n_dim because we summed over n_dim.
        attn_logits = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(n_dim)

        # masking the future with -inf to make probs 0.
        masked_logits = attn_logits + self.mask[:T, :T]

        # Softmax
        max_logit = np.max(masked_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(masked_logits - max_logit)
        attn_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # [B, n_head, T, n_dim] from [B, n_head, T, T] @ [B, n_head, T, n_dim]
        attn_out = np.matmul(attn_probs, v)

        # reshape back to the initial format
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, C)

        output = np.matmul(attn_out, self.w_o)

        self.cache = {
            "x": x,
            "q": q,
            "k": k,
            "v": v,
            "scale": np.sqrt(n_dim),
            "attn_probs": attn_probs,
            "attn_out": attn_out,
        }

        return output

    def backward(self, d_output: np.ndarray):
        x = self.cache["x"]
        B, T, C = x.shape
        n_dim = C // self.n_head
        q = self.cache["q"]  # B, n_head, T, C // n_head
        k = self.cache["k"]  # B, n_head, T, C // n_head
        v = self.cache["v"]  # B, n_head, T, C // n_head
        scale = self.cache["scale"]  # scalar
        attn_out = self.cache["attn_out"]  # B, T, C
        attn_probs = self.cache["attn_probs"]  # B, n_head, T, T

        # We need to do backwards the follow ops:
        # 1st op: out = attn_o @ W_o, we need d_w_o for the gradients and d_attn_out for backprop
        # d_out is [B, T, C]
        # [C, C] comes from [B, C, T] @ [B, T, C], summed over B
        self.d_w_o += np.matmul(attn_out.transpose(0, 2, 1), d_output).sum(axis=0)

        # [B, T, C_in] comes from [B, T, C_out] times [C_out, C_in]
        # (C_in and C_out are the same number but it's good to keep track)
        d_attn_out = np.matmul(d_output, self.w_o.T)

        # d_attn_out is [B, T, C], we need to make it [B, n_head, T, n_dim] to keep propagating:
        d_attn_out = d_attn_out.reshape(B, T, self.n_head, n_dim).transpose(0, 2, 1, 3)

        # Next op: attn_out = attn_probs @ v
        # [B, n_head, T, T] = [B, n_head, T, n_dim_out] @ [B, n_head, n_dim_out, T]
        d_attn_probs = np.matmul(d_attn_out, v.transpose(0, 1, 3, 2))

        # [B, n_head, T, n_dim] = [B, n_head, T, T] @ [B, n_head, T, n_dim]
        d_v = np.matmul(attn_probs.transpose(0, 1, 3, 2), d_attn_out)

        # next op: softmax (the hard one)
        sum_term = np.sum(d_attn_probs * attn_probs, axis=-1, keepdims=True)
        # [B, n_head, T, T] = [B, n_head, T, T] * [B, n_head, T, T]
        d_masked_logits = attn_probs * (d_attn_probs - sum_term)

        # next op: attn_logits = Q@K^T / scale

        # [B, n_head, T, n_dim] = [B, n_head, T, T] @ [B, n_head, T, n_dim]
        d_q = np.matmul(d_masked_logits, k) / scale
        # [B, n_head, n_dim, T] = [B, n_head, n_dim, T] @ [B, n_head, T, T] (then we tranpose because otherwise we calculate d_k.T)
        d_k = (
            np.matmul(q.transpose(0, 1, 3, 2), d_masked_logits).transpose(0, 1, 3, 2)
            / scale
        )

        # we now reshape d_q, d_k and d_v:
        # B, n_head, T, n_dim -> B, T, C
        d_q = d_q.transpose(0, 2, 1, 3).reshape(B, T, C)
        d_k = d_k.transpose(0, 2, 1, 3).reshape(B, T, C)
        d_v = d_v.transpose(0, 2, 1, 3).reshape(B, T, C)

        # Finally x @ W_qkv
        # [B, T, 3 * C]
        d_qkv = np.concatenate([d_q, d_k, d_v], axis=-1)

        # [C, 3 * C] = [B, C, T] @ [B, T, 3 * C] summed over B
        self.d_w_qkv += np.matmul(x.transpose(0, 2, 1), d_qkv).sum(axis=0)

        # To return dx:
        d_x = np.matmul(d_qkv, self.w_qkv.transpose())

        return d_x

    def zero_grad(self):
        self.d_w_qkv.fill(0)
        self.d_w_o.fill(0)

    def step(self, lr):
        # TODO: implement momentum.
        self.w_qkv -= lr * self.d_w_qkv
        self.w_o -= lr * self.d_w_o
