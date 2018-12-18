# from leetcode


def find_maximum_xor(nums):
    max_num = 0
    mask = 0
    for i in range(32)[::-1]:
        mask = mask | (1 << i)
        nums_set = set()
        for num in nums:
            nums_set.add(num & mask)
        tmp = max_num | (1 << i)
        for prefix in nums_set:
            if (tmp ^ prefix) in nums_set:
                max_num = tmp
                break
    return max_num