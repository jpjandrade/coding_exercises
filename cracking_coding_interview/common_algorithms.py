# dfs in tree
from collections import deque


# searching for elements in a tree

def dfs(node, target):
    if not node:
        return None
    print(node.val)
    if node.val == target:
        print("found!")
        return node
    left_node_search = dfs(node.left, target)
    if left_node_search:
        return left_node_search
    return dfs(node.right, target)


# bfs for search binary trees
def bfs(node, target):
    if not node:
        return None

    q = deque()
    q.append(node)
    while len(q) > 0:
        n = q.popleft()
        if n.left:
            q.append(n.left)
        if n.right:
            q.append(n.right)
        print(n.val)
        if n.val == target:
            print("found")
            return n

    return None


# mergesort


def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    cutting_point = len(arr) // 2
    left_sorted = merge_sort(arr[:cutting_point])
    right_sorted = merge_sort(arr[cutting_point:])
    return merge(left_sorted, right_sorted)


def merge(left_arr, right_arr):
    final_arr = []
    left_index = 0
    right_index = 0
    while left_index < len(left_arr) and right_index < len(right_arr):
        if left_arr[left_index] <= right_arr[right_index]:
            final_arr.append(left_arr[left_index])
            left_index += 1
        else:
            final_arr.append(right_arr[right_index])
            right_index += 1

    append_leftovers(final_arr, left_arr, left_index)
    append_leftovers(final_arr, right_arr, right_index)

    return final_arr


def append_leftovers(arr, leftover_arr, curr_index):
    for i in range(curr_index, len(leftover_arr)):
        arr.append(leftover_arr[i])
# quicksort


def quicksort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    lower_than_pivot = [elem for elem in arr if elem < pivot]
    greater_than_pivot = [elem for elem in arr if elem > pivot]
    pivot_array = [elem for elem in arr if elem == pivot]
    return quicksort(lower_than_pivot) + pivot_array + quicksort(greater_than_pivot)


# binary search

def binary_search(arr, t):
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == t:
            return mid
        if arr[mid] < t:
            low = mid + 1
        if arr[mid] > t:
            high = mid - 1

    return -1