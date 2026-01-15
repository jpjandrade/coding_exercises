import numpy as np


class AttentionBlock:
    def __init__(self, T, C, n_heads):
        self.w_q = np.random.randn(C, C)
        self.w_k = np.random.randn(C, C)
        self.w_v = np.random.randn(C, C)
        self.w_o = np.random.randn(C, C)

        self.n_heads = n_heads

        self.mask = np.triu(np.full((T, T), -np.inf), k=1)

    def forward(self, x: np.ndarray):
        B, T, C = x.shape
        n_dim = C // self.n_heads

        # all results are B, n_heads, T, n_dim
        q = np.matmul(x, self.w_q)
        k = np.matmul(x, self.w_k)
        v = np.matmul(x, self.w_v)

        # B, T, n_heads, n_dim -> B, n_heads, T, n_dim
        q = q.reshape(B, T, self.n_heads, n_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, n_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, n_dim).transpose(0, 2, 1, 3)

        # q is B, n_heads, T, n_dim and k is B, n_heads, n_dim, T
        # att_logits is B, n_heads, T, T and we divide over n_dim because we summed over n_dim.
        attn_logits = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(n_dim)

        # masking the future with -inf to make probs 0.
        masked_logits = attn_logits + self.mask

        # Softmax
        max_logit = np.max(masked_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(masked_logits - max_logit)
        attn_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        attn_out = np.matmul(attn_probs, v)

        # reshape back to the initial format
        output = attn_out.transpose(0, 2, 1, 3).reshape(B, T, C)

        output = np.matmul(output, self.w_o)
        return output
