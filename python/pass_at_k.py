import math
from typing import List


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k metric.

    Args:
        n: total number of completions generated
        c: number of completions that passed
        k: k value for pass@k

    Returns:
        Probability that at least one of k random samples passes

    Formula: 1 - (C(n - c, k) / C(n, k))
    Where C(a,b) is "a choose b"
    """
    if n - c < k:
        return float(c > 0)

    return 1 - math.comb(n - c, k) / math.comb(n, k)


def evaluate_pass_at_k(results: List[List[bool]], k: int) -> float:
    """
    Given results for multiple problems, compute average pass@k.

    Args:
        results: List of lists, where results[i] is list of pass/fail
                 for each completion on problem i
        k: k value for pass@k

    Returns:
        Average pass@k across all problems
    """
    total = 0
    for result in results:
        n = len(result)
        c = sum(result)
        total += pass_at_k(n, c, k)
    return total / len(results)


# Test cases
# 10 completions, 3 passed, k=1: what's probability at least 1 of 1 sample passes?
assert abs(pass_at_k(10, 3, 1) - 0.3) < 0.01

# 10 completions, 3 passed, k=10: must include a passing one
assert pass_at_k(10, 3, 10) == 1.0

# 10 completions, 0 passed: always 0
assert pass_at_k(10, 0, 5) == 0.0

# Mock results for 3 problems
results = [
    [True, False, False, True, False],  # Problem 0: 2/5 passed
    [False, False, False, False, False],  # Problem 1: 0/5 passed
    [True, True, True, False, False],  # Problem 2: 3/5 passed
]
# Your function should compute average pass@k across these

for k in range(len(results[0]) + 1):
    print(evaluate_pass_at_k(results, k))
