public class ListNode {
    ListNode next = null;
    int data;

    public ListNode(int d) {
        data = d;
    }

    void appendToTail(int d) {
        ListNode end = new ListNode(d);
        ListNode n = this;
        while (n.next != null) {
            n = n.next;
        }
        n.next = end;
    }

    ListNode deleteNode(ListNode head, int d){
        ListNode n = head;

        if (n.data == d) {
            return head.next;
        }

        while (n.next != null) {
            if (n.next.data == d) {
                n.next = n.next.next;
                return head;
            }
            n = n.next;
        }

        return head;
    }
}
