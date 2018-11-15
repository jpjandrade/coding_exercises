# merge sort as seen in the book, implemented in python by me


def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    middle_point = len(arr) // 2
    left_arr = arr[0:middle_point]
    right_arr = arr[middle_point:]
    left_sorted = merge_sort(left_arr)
    right_sorted = merge_sort(right_arr)
    return merge_sorted_arrays(left_sorted, right_sorted)


def merge_sorted_arrays(left_arr, right_arr):
    final_arr = left_arr + right_arr
    left_index = 0
    right_index = 0
    current_index = 0
    while left_index < len(left_arr) and right_index < len(right_arr):
        if left_arr[left_index] <= right_arr[right_index]:
            final_arr[current_index] = left_arr[left_index]
            left_index += 1
        else:
            final_arr[current_index] = right_arr[right_index]
            right_index += 1
        current_index += 1

    remaining_left_entries = len(left_arr) - left_index
    for i in range(remaining_left_entries):
        final_arr[current_index + i] = left_arr[left_index + i]

    return final_arr


def quick_sort(arr):
    if arr is None or len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]  # O(1)
    left = [elem for elem in arr if elem < pivot]  # O(n)
    right = [elem for elem in arr if elem > pivot]  # O(n)
    all_pivots = [elem for elem in arr if elem == pivot]  # because life sucks and sometimes pivot has multiples
    return quick_sort(left) + all_pivots + quick_sort(right)  # will be called O(log n) times


def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    middle = len(arr) // 2
    while low <= high:
        middle = (low + high) // 2
        if arr[middle] > x:
            high = middle - 1
        elif arr[middle] < x:
            low = middle + 1
        else:
            return middle
    return -1


# ex 10.1
# while the exercise gives room for a solution which is O(1) in space,
# I'm going for something which is O(n) in space and speed
def sorted_merge(arr_a, arr_b):
    final_arr = [None for elem in arr_a + arr_b]  # in theory arr_a is enough but no such thing in python
    arr_a = arr_a[:]  # let's not destroy arrays outside the function
    arr_b = arr_b[:]
    idx_a = 0
    idx_b = 0
    max_elem = max(arr_a[-1], arr_b[-1]) + 1 # both arrays sorted

    # helper elements so we're able to freely compare without index issues
    arr_a.append(max_elem)
    arr_b.append(max_elem)

    for i in range(len(final_arr)):  # will not have the extra elements
        if arr_a[idx_a] < arr_b[idx_b]:
            final_arr[i] = arr_a[idx_a]
            idx_a += 1
        else:
            final_arr[i] = arr_b[idx_b]
            idx_b += 1

    return final_arr

# ex 10.1 after looking up hint, fine, let's do it O(1) in space


def sorted_merge_efficient(arr_a, arr_b):
    arr_a = arr_a + [None for elem in arr_b]  # sorry, it's still python
    current_index = len(arr_a) - 1
    idx_a = -1
    for elem in arr_a:  # not assuming arr_a doesn't have extra space
        if elem is not None:
            idx_a += 1
    idx_b = len(arr_b) - 1
    while idx_b >= 0:
        if idx_a >= 0 and arr_a[idx_a] > arr_b[idx_b]:
            arr_a[current_index] = arr_a[idx_a]
            idx_a -= 1
        else:
            arr_a[current_index] = arr_b[idx_b]
            idx_b -= 1
        current_index -= 1
    return arr_a


# ex 10.2

def sort_string(s):  # python's built in function sorts but not sure if can use it
    str_as_ord = [ord(c) for c in s]
    sorted_ordinals = quick_sort(str_as_ord)
    return ''.join([chr(n) for n in sorted_ordinals])


def return_anagrams(arr):
    all_words = {}
    for word in arr:
        sorted_word = sort_string(word)
        if sorted_word in all_words:
            all_words[sorted_word].append(word)
        else:
            all_words[sorted_word] = [word]
    grouped_anagrams = []
    for sorted_word in all_words:
        grouped_anagrams += all_words[sorted_word]
    return grouped_anagrams

# ex 10.7
# yolo solution, O(n log n), just search for every int between 0 and four billion :-P
# O(1) in memory btw. Keep your 1GB / 10 MB :-P

def missing_int_yolo(big_ass_array):
    for i in range(len(big_ass_array)):
        if binary_search(big_ass_array, i) == -1:
            return i
    return len(big_ass_array)  # I mean


# turns out yolo solution doesn't fit in memory unless you have a badass computer
# the part about reading from a file was very relevant :-P
# I'll keep the above as a memento from my naivety
