
# 1.1


def check_string(s):  # O(n)

    seen_count = {}
    for char in s:
        if char in seen_count:
            return False
        else:
            seen_count[char] = 1
    return True


def check_string_no_hash(s):  # this is O(n^2)
    for i in len(s):
        c = s[i]
        for j in range(i):
            if s[j] == c:
                return False

    return True

# 1.2


def check_if_permutation(s1, s2):
    """
    Assumes no extraneous whitespace (i.e., "Smith  " not equal to "Smith")
    """
    ALPHABET_LENGTH = 128  # arbitrary

    if len(s1) != len(s2):
        return False

    char_count = [0 for i in range(ALPHABET_LENGTH)]

    for c in s1:
        char_count[ord(c) - ord('A')] += 1
    for c in s2:
        char_count[ord(c) - ord('A')] -= 1

    for count in char_count:
        if count != 0:
            return False

    return True
