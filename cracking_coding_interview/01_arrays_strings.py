
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

def check_if_palindrome(s):
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


# 1.5

def check_one_edit(s1, s2):
    if abs(len(s1) - len(s2)) > 1:
        return False
    elif abs(len(s1) - len(s2)) == 1:
        return check_for_insert(s1, s2)
    else:
        return check_for_replace(s1, s2)


def check_for_replace(s1, s2):
    assert(len(s1) == len(s2))  # just to be sure :-P
    edit_found = False
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            if edit_found:
                return False
            else:
                edit_found = True

    return True


def check_for_insert(s1, s2):
    if len(s1) > len(s2):
        g = s1
        l = s2
    else:
        g = s2
        l = s1

    idx = -1
    insert_found = False
    for i in range(len(l)):
        idx += 1
        if g[idx] != l[i]:
            if insert_found:
                return False
            else:
                insert_found = True
                idx += 1
                if g[idx] != l[i]:
                    return False

    return True


# ex 1.6

def compress_string(s):
    compressed = ['' for i in range(2 * len(s))]  # worst case scenario
    curr_char = s[0]
    curr_count = 1
    idx = 0
    for i in range(1, len(s)):
        c = s[i]
        if c == curr_char:
            curr_count += 1
        else:
            compressed[idx] = curr_char
            compressed[idx + 1] = str(curr_count)
            idx += 2
            curr_char = c
            curr_count = 1

    compressed[idx] = curr_char
    compressed[idx + 1] = str(curr_count)

    compressed_string = ''.join(compressed)

    if len(compressed_string) < len(s):
        return compressed_string
    else:
        return s
