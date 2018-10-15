class Node:
    def __init__(self, d):
        self.data = d
        self.next = None


class Stack:
    def __init__(self):
        self.top = None

    def pop(self):
        if self.top is None:
            raise AttributeError  # I didn't know which error to put here :-P
        item = self.top.data
        self.top = self.top.next
        return item

    def push(self, item):
        t = Node(item)
        t.next = self.top
        self.top = t

    def peek(self):
        if self.top is None:
            raise AttributeError

        return self.top.data

    def is_empty(self):
        return self.top is None


class Queue:
    def __init__(self):
        self.first = None
        self.last = None

    def add(self, data):
        t = Node(data)
        if self.last is not None:
            self.last.next = t
        self.last = t
        if self.first is None:
            self.first = self.last

    def remove(self):
        if self.first is None:
            raise EOFError  # trying random errors :-P

        item = self.first.data
        self.first = self.first.next

        if self.first is None:
            self.last = None

        return item

    def peek(self):
        if self.first is None:
            raise EOFError
        return self.first.data

    def is_empty(self):
        return self.first is None


# ex 3.2

class StackMin(Stack):
    def __init__(self):
        super().__init__()
        self.min_stack = Stack()

    def pop(self):
        item = super().pop()
        if item == self.min_stack.peek():
            self.min_stack.pop()
        return item

    def push(self, item):
        super().push(item)
        if self.min_stack.is_empty() or item <= self.min():
            self.min_stack.push(item)

    def min(self):
        return self.min_stack.peek()

# 3.3

class SetOfStacks:

    def __init__(self):
        self.MAX_STACKS = 3
        self.MAX_SIZE = 2

        self.all_stacks = [Stack() for i in range(self.MAX_STACKS)]
        self.current_stack = 0
        self.stack_sizes = [0 for i in range(self.MAX_STACKS)]
        self.all_full = False

    def pop(self):
        s = self.all_stacks[self.current_stack]
        item = s.pop()
        self.stack_sizes[self.current_stack] -= 1
        if s.is_empty() and self.current_stack > 0:
            self.current_stack -= 1
        if self.all_full:
            self.all_full = False
        return item

    def push(self, item):
        if self.all_full:
            raise ValueError("All stacks full!")
        s = self.all_stacks[self.current_stack]
        s.push(item)
        self.stack_sizes[self.current_stack] += 1
        while not self.all_full and self.stack_sizes[self.current_stack] == self.MAX_SIZE:
            self.current_stack += 1
            if self.current_stack == self.MAX_STACKS:
                self.all_full = True

    def peek(self):
        s = self.all_stacks[self.current_stack]
        item = s.peek()
        return item

    def pop_at(self, index):
        s = self.all_stacks[index]
        item = s.pop()
        self.stack_sizes[index] -= 1
        if self.current_stack > index:
            self.current_stack = index
        if self.all_full:
            self.all_full = False
        return item