import random
from ch_02_linked_lists import ListNode
from ch_03_stacks_queues import Queue

# binary tree definitions


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


def in_order_traversal(node, visit):
    if node is not None:
        in_order_traversal(node.left, visit)
        visit(node)
        in_order_traversal(node.right, visit)


def pre_order_traversal(node, visit):
    if node is not None:
        visit(node)
        pre_order_traversal(node.left, visit)
        pre_order_traversal(node.right, visit)


def post_order_traversal(node, visit):
    if node is not None:
        post_order_traversal(node.left, visit)
        post_order_traversal(node.right, visit)
        visit(node)


# graph definitions


class GraphNode:
    def __init__(self, d):
        self.data = d
        self.connections = []


class Graph:
    def __init__(self):
        self.nodes = []


def depth_first_search(node, visit):
    if node is not None:
        visit(node)
        node.visited = True
        for n in node.connections:
            if hasattr(n, 'visited') is False:  # whoo duck typing fuck yeah ᕕ( ᐛ )ᕗ
                depth_first_search(n, visit)


def breadth_first_search(node, visit):
    queue = Queue()
    node.queued = True
    queue.add(node)

    while not queue.is_empty():
        node = queue.remove()
        visit(node)
        for n in node.connections:
            if hasattr(n, 'queued') is False:
                queue.add(n)
                n.queued = True

# auxiliary functions for checking, not part of exercises


def generate_random_tree(max_depth, curr_depth):
    if curr_depth > max_depth:
        return None
    else:
        d = random.randint(-3, 7)
        leaf = BinaryNode(d)
        leaf.left = generate_random_tree(max_depth, curr_depth + 1)
        leaf.right = generate_random_tree(max_depth, curr_depth + 1)
        return leaf


def generate_random_graph(num_nodes, index_as_data=True):
    g = Graph()
    for i in range(num_nodes):
        if index_as_data:
            data = i
        else:
            data = random.randint(-2, 6)
        g.nodes.append(GraphNode(data))
    for i in range(num_nodes):
        node = g.nodes[i]
        n_connections = random.randint(0, num_nodes - 1)
        possible_nodes = [j for j in range(num_nodes) if j != i]
        node.connections = [g.nodes[j] for j in random.sample(possible_nodes, n_connections)]
    return g


t = generate_random_tree(3, 0)
g = generate_random_graph(5)


def max_with_none(*arr):
    filtered_arr = [elem for elem in arr if elem is not None]
    return max(filtered_arr)


def max_of_tree(t):
    if t is None:
        return None
    return max_with_none(t.data, max_of_tree(t.left), max_of_tree(t.right))


# ex 4.1
# I will assume all nodes have distinct data so we are looking for a node with a certain data
# this is easily extendable if we need to: we can have an node.name or node.index property

def bfs_with_break(node, predicate):
    q = Queue()
    q.add(node)
    node.queued = True

    while not q.is_empty():
        node = q.remove()
        if predicate(node):
            return True
        else:
            for n in node.connections:
                if hasattr(n, 'queued') == False:
                    q.add(n)
                    n.queued = True

    return False


def find_path(n1, n2):
    p1 = lambda x: x.data == n1.data
    p2 = lambda x: x.data == n2.data

    return bfs_with_break(n1, p2) or bfs_with_break(n2, p1)


# ex 4.2

# after looking at the solution I realize by slicing the array
# every time I added a O(n) step to each iteration.
# I'll leave the function as is but it wasn't necessary :-(

def split_array(arr, arr_len):
    split_results = {}
    midpoint = (arr_len - 1) // 2
    split_results['node'] = arr[midpoint]
    split_results['left_arr'] = arr[:midpoint]
    split_results['left_len'] = (arr_len - 1) // 2  # biasing towards left being larger
    split_results['right_arr'] = arr[midpoint + 1:]
    split_results['right_len'] = arr_len // 2

    return split_results


def create_minimal_tree(arr, arr_len):
    if arr_len == 0:
        return None
    if arr_len == 1:
        n = BinaryNode(arr[0])
    else:
        split = split_array(arr, arr_len)
        n = BinaryNode(split['node'])
        n.left = create_minimal_tree(split['left_arr'], split['left_len'])
        n.right = create_minimal_tree(split['right_arr'], split['right_len'])
    return n


def create_minimal_tree_root(sorted_arr):
    N = len(sorted_arr)
    root = create_minimal_tree(sorted_arr, N)
    return root


# ex 4.3

# I decided to implement the "list of list" element as an array
# and I do "append" at every new depth level. This is inefficient
# and I could get better performance by doing a linked list at that
# level as well.

# edit: after looking up the solution I totally overengineered this :-P
# but I do believe it still runs at the same time and uses the same (big O) space
# so I'll leave it as is

def num_of_children(node):
    has_left = node.left is not None
    has_right = node.right is not None
    return int(has_left) + int(has_right)


def nodes_per_depth(node):
    all_lists = []
    if node is None:
        return all_lists
    q = Queue()
    q.add(node)
    count = 0
    level = 0
    curr_max = 1
    next_max = 0
    all_lists.append(None)
    while not q.is_empty():
        n = q.remove()
        curr_list = all_lists[level]
        new_list = ListNode(n.data)
        new_list.next = curr_list
        all_lists[level] = new_list
        next_max += num_of_children(n)
        count += 1
        if n.left is not None:
            q.add(n.left)
        if n.right is not None:
            q.add(n.right)
        if count >= curr_max:
            curr_max = next_max
            next_max = 0
            count = 0
            level += 1
            if not q.is_empty():
                all_lists.append(None)

    return all_lists


# ex 4.4
# this is super not clean code because it's a function that does two things at once. But it runs
# in O(n) so yay? Also if the implicit tuple expansion is a problem we can use dicts

# edit: after looking at the solution, I prefer my tuple usage better than overloading the height variable :-)
def balance_and_heigt(node):
    if node is None:
        return True, 0
    left_balance, left_height = balance_and_heigt(node.left)
    right_balance, right_height = balance_and_heigt(node.right)

    balanced = left_balance and right_balance and abs(left_height - right_height) <= 1
    height = 1 + max(left_height, right_height)
    return balanced, height


def is_balanced(node):
    balanced, _ = balance_and_heigt(node)
    return balanced

# ex 4.5
# totally got this wrong after reading solution :-(. Rewrote the book's solution in python only


def check_if_node_between_min_max(node, curr_min, curr_max):
    if node is None:
        return True
    if (curr_min is not None and node.data <= curr_min) or (curr_max is not None and node.data > curr_max):
        return False
    if not check_if_node_between_min_max(node.left, curr_min, node.data) or not check_if_node_between_min_max(node.right, node.data, curr_max):
        return False
    return True


def check_if_bst(root):
    return check_if_node_between_min_max(root, None, None)


# ex 4.6
# I did this one a bit different by checking the parents values explicitely. I believe end result is the same
def find_min_of_bst(node):
    if node is None or node.left is None:
        return node
    else:
        return find_min_of_bst(node.left)


def find_first_larger_parent(node, d):
    n = node
    while n.parent is not None:
        n = n.parent
        if n.data >= d:
            return n

    raise EOFError


def find_successor(node):
    if node.right is not None:
        return find_min_of_bst(node.right)
    else:
        try:
            return find_first_larger_parent(node, node.data)
        except EOFError:
            return None


# ex 4.8
# so many corner cases this solution must be stupid

def find_common_ancestor(root, node_a, node_b):
    path_to_a = find_path_on_binary_tree(root, node_a, [])
    path_to_b = find_path_on_binary_tree(root, node_b, [])
    if path_to_a is None and path_to_b is None:
        return None
    if path_to_a is None:
        return rebuild_path(path_to_b)
    if path_to_b is None:
        return rebuild_path(path_to_a)

    divergence_idx = find_path_divergence(path_to_a, path_to_b)
    ancestor = rebuild_path(root, path_to_a, divergence_idx)
    return ancestor


def find_path_on_binary_tree(node, t, curr_path):
    if not node:
        return None
    if node.data == t:
        return curr_path
    left_path = find_path_on_binary_tree(node.left, t, curr_path + ['l'])
    if left_path:
        return left_path
    return find_path_on_binary_tree(node.right, t, curr_path + ['r'])


def find_path_divergence(path_a, path_b):
    min_len = min(len(path_a), len(path_b))
    if min_len == 0:
        return 0
    for i in range(min_len):
        if path_a[i] != path_b[i]:
            return i
    return min_len


def rebuild_path(root, path, idx):
    if idx is None or idx < 0:
        raise ValueError("No common path")
    common_path = path[:idx]  # exclusive of idx
    node = root
    for direction in common_path:
        if direction == 'r':
            node = node.right
        else:
            node = node.left
    return node

