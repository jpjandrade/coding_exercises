package LeetCode;

public class Lc148SortList {
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode mid = splitMid(head);
        head = sortList(head);
        mid = sortList(mid);
        return merge(head, mid);
    }

    public ListNode splitMid(ListNode head) {
        ListNode fast = head;
        ListNode prev = null;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            prev = head;
            head = head.next;
        }
        if (prev != null) {
            prev.next = null;
        }
        return head;
    }

    public ListNode merge(ListNode left, ListNode right) {
        ListNode sentinel = new ListNode(-1);
        ListNode head = sentinel;
        while (left != null && right != null) {
            if (left.val < right.val) {
                head.next = left;
                left = left.next;
            } else {
                head.next = right;
                right = right.next;
            }
            head = head.next;
        }

        while (left != null) {
            head.next = left;
            left = left.next;
            head = head.next;
        }
        while (right != null) {
            head.next = right;
            right = right.next;
            head = head.next;
        }

        return sentinel.next;
    }
}
