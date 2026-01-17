import numpy as np
from attention import CausalSelfAttention
from mlp import MLP


def numerical_gradient(f, x, eps=1e-5):
    """Compute numerical gradient for verification."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        x[idx] = old_val + eps
        pos = f().copy()

        x[idx] = old_val - eps
        neg = f().copy()

        x[idx] = old_val
        grad[idx] = np.sum(pos - neg) / (2 * eps)
        it.iternext()
    return grad


def validate_gradients(layer, x):
    B, T, C = x.shape

    # Forward pass
    out = layer.forward(x)

    # Assume loss = sum(out), so d_out = ones
    d_out = np.ones_like(out)

    # Backward pass
    d_x = layer.backward(d_out)

    # Verify with numerical gradients
    def loss_fn():
        return layer.forward(x)

    d_x_numerical = numerical_gradient(loss_fn, x)

    print("Analytical vs Numerical gradient check for d_x:")
    print(f"  Max absolute difference: {np.max(np.abs(d_x - d_x_numerical)):.2e}")
    print(
        f"  Relative error: {np.linalg.norm(d_x - d_x_numerical) / (np.linalg.norm(d_x) + 1e-8):.2e}"
    )

    # Check weight gradients too
    def loss_fn_w():
        return layer.forward(x)

    for weight_name in layer.weights_with_grad:
        w, d_w = layer.weights_with_grad[weight_name]
        dw_numerical = numerical_gradient(loss_fn_w, w)
        print(f"\nAnalytical vs Numerical gradient check for {weight_name}:")
        print(f"  Max absolute difference: {np.max(np.abs(d_w - dw_numerical)):.2e}")
        print(
            f"  Relative error: {np.linalg.norm(d_w - dw_numerical) / (np.linalg.norm(d_w) + 1e-8):.2e}"
        )


def main():
    np.random.seed(42)
    B, T, C, n_heads = 2, 4, 8, 2
    x = np.random.randn(B, T, C)
    layers = [CausalSelfAttention(T, C, n_heads), MLP(C)]
    for layer in layers:
        validate_gradients(layer, x)


if __name__ == "__main__":
    main()
