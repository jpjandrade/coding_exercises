# first an implementation of a linked list in python for practicing examples


class ListNode:
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
        end = ListNode(d)
        n = self
        while n.next is not None:
            n = n.next

        n.next = end


def delete_value(head, d):
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


# 2.2
# I assume k = 1 means last entry
def kth_to_last(head, k):
    count = 0
    n = head
    delayed_head = None
    if k == 1 and head.next is None:  # whoo corner cases!
        return head.data
    while n.next is not None:
        if count == k - 1:
            delayed_head = head
        n = n.next
        count += 1
        if delayed_head is not None:
            delayed_head = delayed_head.next

    if count == k - 1:
            delayed_head = head

    if delayed_head is not None:
        return delayed_head.data
    else:
        raise ValueError("k is larger than the list")


# 2.3
# I didn't check for data or next being None because exercise says it isn't the first or last
def delete_this_node(head):
    head.data = head.next.data
    head.next = head.next.next
