package LeetCode;

public class Lc876MiddleOfLinkedList {
    public ListNode middleNode(ListNode head) {
        ListNode fast = head;

        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            head = head.next;
        }

        return head;
    }
}
