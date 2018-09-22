
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
    for i in range(len(s)):
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


# 1.3
# we'll only strip the spaces in the end of the string

def urlify(s):
    last_idx = len(s) - 1

    while s[last_idx] == ' ' and last_idx > 0:
        last_idx -= 1
    effective_length = last_idx + 1

    spaces_count = 0
    for i in range(effective_length):  # O(n)
        if s[i] == ' ':
            spaces_count += 1

    new_string = [None for i in range(effective_length + 2 * spaces_count)]
    idx = 0
    for i in range(effective_length):
        if s[i] == ' ':
            new_string[idx] = '%'
            new_string[idx + 1] = '2'
            new_string[idx + 2] = '0'
            idx += 3
        else:
            new_string[idx] = s[i]
            idx += 1
    final_string = ''.join(new_string)
    return final_string


# 1.4

def palindrome_checker(s):
    char_count = {}
    for c in s:
        if c != ' ':
            if c in char_count:
                char_count[c] += 1
            else:
                char_count[c] = 1

    even = 0
    odds = 0

    for c in char_count:
        if char_count[c] % 2 == 0:
            even += 1
        else:
            odds += 1

    if odds >= 1:
        return True
    else:
        return False
