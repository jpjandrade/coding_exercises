def edit_distance(word_1: str, word_2: str) -> int:
    # Calculates the number of insertions, deletions or replacements
    # needed to make word_1 into word_2 and vice-versa.
    # Equivalent to https://leetcode.com/problems/edit-distance.

    if not word_1 and not word_2:
        return 0

    n = len(word_1)
    m = len(word_2)

    if not word_1 or not word_2:
        return max(n, m)

    # dp[0, 0] is the edit distance between empty strings.
    # dp uses 1-based indexing: dp[i+1][j+1] represents edit distance between word_1[:i+1] and word_2[:j+1]
    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    # Pure insert distance from empty string to word_1.
    for i in range(n):
        dp[i + 1][0] = i + 1

    # Pure insert distance from empty string to word_2.
    for j in range(m):
        dp[0][j + 1] = j + 1

    for i in range(n):
        for j in range(m):
            if word_1[i] == word_2[j]:
                # if characters are the same, then we just take the edit distance from the i - 1, j - 1 strings.
                dp[i + 1][j + 1] = dp[i][j]
            else:
                # We either insert / delete from one char off or replace from i - 1, j - 1.
                dp[i + 1][j + 1] = 1 + min(dp[i + 1][j], dp[i][j + 1], dp[i][j])

    return dp[-1][-1]


def survival_score(input_string: str, output_string: str) -> float:
    """
    Compute fraction of how much of the original string 'survived' in the output one.
    Use Levenshtein edit distance, normalized by the longer string.

    Returns: 1.0 if identical, 0.0 if completely different
    """
    if not input_string:
        return 1.0  # Nothing to lose

    if not output_string:
        return 0.0  # Everything lost

    dist = edit_distance(input_string, output_string)
    max_length = max(len(input_string), len(output_string))
    return 1 - dist / max_length


# Test cases
assert survival_score("return a + b", "return a + b") == 1.0
assert survival_score("", "") == 1.0
assert survival_score("abc", "xyz") == 0.0
assert 0.5 < survival_score("return a + b", "return a + b + c") < 1.0

print("All tests passed.")
