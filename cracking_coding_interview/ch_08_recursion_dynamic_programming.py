# ex 8.1


def stairs_comb(n):
    memo = []
    memo.append(0)  # appending is less explicit but takes care of corner cases (i.e., n = 1)
    memo.append(1)
    memo.append(2)
    memo.append(4)
    if n >= 4:
        for i in range(4, n + 1):
            combinations = memo[i - 1] + memo[i - 2] + memo[i - 3]
            memo.append(combinations)
    return memo[n]