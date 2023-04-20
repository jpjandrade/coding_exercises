import math

import torch
from torch import nn


class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def _masked_softmax(self, scores, valid_lens):
        """
        Perform softmax operation by masking elements on the last axis.
        """
        # `scores`: 3D tensor, `valid_lens`: 1D or 2D tensor
        if valid_lens is None:
            return nn.functional.softmax(scores, dim=-1)

        else:
            shape = scores.shape
            if valid_lens.dim() == 1:
                valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            else:
                valid_lens = valid_lens.reshape(-1)
            # On the last axis, replace masked elements with a very large negative
            # value, whose exponentiation outputs 0
            scores = self._sequence_mask(
                scores.reshape(-1, shape[-1]), valid_lens, value=-1e6
            )
            return nn.functional.softmax(scores.reshape(shape), dim=-1)

    def _sequence_mask(self, X, valid_len, value=-1e6):
        """
        Masks entries in the sequence which are beyond its valid_len.
        e.g., a 17 token sentence will have max_len - 17 irrelevant entries in its tensor.
        We replace these by `value` (usually a very high negative number so that softmax makes it zero)
        """
        max_len = X.size(1)
        mask = (
            torch.arange((max_len), dtype=torch.float32, device=X.device)[None, :]
            < valid_len[:, None]
        )
        X[~mask] = value
        return X

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = self._masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        num_heads,
        dropout,
        bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def _transpose_qkv(self, X):
        # Shape of input `X`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
        # Shape of output `X`:
        # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
        # `num_hiddens` / `num_heads`)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)

        # Shape of output `X`:
        # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        X = X.permute(0, 2, 1, 3)

        # Shape of `output`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        X = X.reshape(-1, X.shape[2], X.shape[3])
        return X

    def _transpose_output(self, X):
        """
        Reverses the operation of `transpose_qkv`
        """
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, queries, keys, values, valid_lens):
        """
        Parameters:
        queries / keys / values: (batch_size, q / k / v, num_hiddens)-dim tensor
        valid_lens = (batch_size, q)-dim tensor
        """

        # tranpose_qkv makes the (B, max_len, num_hiddens) tensor into a
        # (B * num_heads, max_len, num_hiddes // num_heads) tensor, decreasing
        # the embedding dimension but having more independent batches for multi_head
        # attention
        queries = self._transpose_qkv(self.W_q(queries))
        keys = self._transpose_qkv(self.W_k(keys))
        values = self._transpose_qkv(self.W_k(values))

        if valid_lens is not None:
            # repeats each number of the valid_lens tensor num_heads times.
            # so [0, 1, 2] becomes [0, 0, 0, 1, 1, 1, 2, 2, 2] for num_heads=3
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = self._transpose_output(output)

        return self.W_o(output_concat)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        normalized_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        dropout,
        use_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias
        )
        self.add_norm1 = AddNorm(normalized_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.add_norm2 = AddNorm(normalized_shape, dropout)

    def forward(self, X, valid_lens):
        # input X is (B, max_len, num_hiddens)
        # output of attention is (B, max_len, num_hiddens)
        Y = self.add_norm1(X, self.attention(X, X, X, valid_lens))
        return self.add_norm2(Y, self.ffn(Y))


class BERTEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        max_len=1000,
        key_size=768,
        query_size=768,
        value_size=768,
        **kwargs,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(
                f"{i}",
                EncoderBlock(
                    key_size,
                    query_size,
                    value_size,
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    use_bias=True,
                ),
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # tokens and segments are both (B, max_len), when passed through the embeddings
        # layer they become (B, max_len, num_hiddens)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)

        # pos_embedding.data is (1, max_len, num_hiddens), so the sum is broadcast
        # the [:, : X.shape[1], :] block is in case the dataset was created with a
        # different max_len than BERTEncoder
        X = X + self.pos_embedding.data[:, : X.shape[1], :]

        for block in self.blocks:
            X = block(X, valid_lens)

        return X


class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, vocab_size),
        )

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)

        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePred(nn.Module):
    def __init__(self, num_inputs, **kwargs):
        super().__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)


class BERTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        max_len=1000,
        key_size=768,
        query_size=768,
        value_size=768,
        hid_in_features=768,
        mlm_in_features=768,
        nsp_in_features=768,
    ):
        super().__init__()
        self.encoder = BERTEncoder(
            vocab_size,
            num_hiddens,
            norm_shape,
            ffn_num_input,
            ffn_num_hiddens,
            num_heads,
            num_layers,
            dropout,
            max_len=max_len,
            key_size=key_size,
            query_size=query_size,
            value_size=value_size,
        )
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens), nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None

        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
