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
