public class ListNode <T> {
    ListNode<T> next = null;
    T data;

    public ListNode(T d) {
        data = d;
    }

    public void appendToTail(T d) {
        ListNode<T> end = new ListNode<T>(d);
        ListNode<T> n = this;
        while (n.next != null) {
            n = n.next;
        }
        n.next = end;
    }

    public ListNode<T> deleteNode(ListNode<T> head, T d){
        ListNode<T> n = head;

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

    public void print() {
        ListNode<T> n = this;
        System.out.print(this.data);
        while (n.next != null) {
            System.out.print(" -> ");
            System.out.print(n.next.data);
            n = n.next;
        }
        System.out.print("\n");
    }
}
