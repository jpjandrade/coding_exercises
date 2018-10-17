import random


class BinaryNode:
    def __init__(self, d):
        self.data = d
        self.left = None
        self.right = None

    def get_node_str(self, node):
        if node is None:
            return "None"
        else:
            return str(node.data)

    def __str__(self):
        s = "[" + self.get_node_str(self) + ", "
        if self.left is not None or self.right is not None:
            s += "[l: " + str(self.left) + ", r: " + str(self.right) + "]"
        s += "]"
        return s

    def __repr__(self):
        return str(self)


class GraphNode:
    def __init__(self, d):
        self.data = d
        self.connections = []


class Graph:
    def __init__(self):
        self.nodes = []


def in_order_traversal(node, func):
    if node is not None:
        in_order_traversal(node.left, func)
        func(node.data)
        in_order_traversal(node.right, func)


def pre_order_traversal(node, func):
    if node is not None:
        func(node.data)
        pre_order_traversal(node.left, func)
        pre_order_traversal(node.right, func)


def post_order_traversal(node, func):
    if node is not None:
        post_order_traversal(node.left, func)
        post_order_traversal(node.right, func)
        func(node.data)


# not part of any exercise, I just thought I needed a function to generate trees for me :-P

def generate_random_tree(max_depth, curr_depth):
    if curr_depth > max_depth:
        return None
    else:
        d = random.randint(-3, 7)
        leaf = BinaryNode(d)
        leaf.left = generate_random_tree(max_depth, curr_depth + 1)
        leaf.right = generate_random_tree(max_depth, curr_depth + 1)
        return leaf


t = generate_random_tree(3, 0)
