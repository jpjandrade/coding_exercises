# first an implementation of a linked list in python for practicing examples


class Node:
    def __init__(self, d):
        self.data = d
        self.next = None

    # using string concatenation is cheating but this exercise isn't about arrays :-P
    def __str__(self):
        s = '['
        n = self
        s += str(n.data)
        while n.next is not None:
            n = n.next
            s += ", {}".format(n.data)
        s += "]"

        return s

    def __repr__(self):
        return str(self)

    def append_to_tail(self, d):
        end = Node(d)
        n = self
        while n.next is not None:
            n = n.next

        n.next = end


def delete_node(head, d):
    n = head

    if n.data == d:
        return head.next

    while n.next is not None:
        if n.next.data == d:
            n.next = n.next.next
            return head

        n = n.next

    return head


# 2.1

def remove_dups(head):
    seen = {}
    seen[head.data] = 1
    n = head
    while n.next is not None:
        if n.next.data in seen:
            n.next = n.next.next
        else:
            n = n.next
            seen[n.data] = 1

    return head


def delete_nodes(head, d):
    n = head
    while n.next is not None:
        if n.next.data == d:
            n.next = n.next.next
        else:
            n = n.next

    if head.data == d:
        return head.next
    else:
        return head


def remove_dups_no_hash(head):
    n = head
    while n.next is not None:
        n.next = delete_nodes(n.next, n.data)
        n = n.next
    return head
