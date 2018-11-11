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


