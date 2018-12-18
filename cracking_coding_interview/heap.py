class BinHeap:
    def __init__(self):
        self.heap_arr = [0]
        self.current_size = 0

    def bubble_up(self, i):
        while i // 2 > 0:
            if self.heap_arr[i] < self.heap_arr[i // 2]:  # parent is larger than child
                tmp = self.heap_arr[i // 2]
                self.heap_arr[i // 2] = self.heap_arr[i]
                self.heap_arr[i] = tmp
                i = i // 2

    def insert(self, k):
        self.heap_arr.append(k)
        self.current_size = self.current_size + 1
        self.bubble_up(self.current_size)

    def bubble_down(self, i):
        while (i * 2) <= self.current_size:
            mc = self.min_child(i)
            if self.heap_arr[i] > self.heap_arr[mc]:
                tmp = self.heap_arr[i]
                self.heap_arr[i] = self.heap_arr[mc]
                self.heap_arr[mc] = tmp
            i = mc

    def min_child(self, i):
        if i * 2 + 1 > self.current_size:
            return i * 2
        else:
            if self.heap_arr[i * 2] < self.heap_arr[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1

    def delete_min(self):
        item = self.heap_arr[1]
        self.heap_arr[1] = self.heap_arr[self.current_size]
        self.current_size = self.current_size - 1
        self.heap_arr.pop()
        self.bubble_down(1)
        return item

    def build_heap(self, arr):
        i = len(arr) // 2
        self.current_size = len(arr)
        self.heap_arr = [0] + arr[:]
        while (i > 0):
            self.bubble_down(i)
            i = i - 1