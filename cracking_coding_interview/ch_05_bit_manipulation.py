def get_bit(num, i):
    """
    Gets the value of bit in the poition i for number num
    """
    return int((num & (1 << i)) != 0)  # parentheses are not necessary due to precedence


def set_bit(num, i):
    """
    Sets the bit at poistion i to 1
    """
    return num | (1 << i)  # parentheses are not necessary due to precedence


def clear_bit(num, i):
    mask = ~(1 << i)  # something like 11110111
    return num & mask


def update_bit(num, i, bit_is_1):
    value = 1 if bit_is_1 else 0
    mask = ~(1 << i)  # something like 11110111
    return (num & mask) | (value << i)  # clears the bit and upates the value


# ex 5.1
def bit_insertion(N, M, i, j):
    for pos in range(i, j + 1):
        b = get_bit(M, pos - i)
        N = update_bit(N, pos, b)
    return N
