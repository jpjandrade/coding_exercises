import java.util.ArrayList;
import java.util.HashSet;

public class LinkedLists {

    public static <T> ListNode<T> buildLinkedList(T[] elements) {
        if (elements.length == 0) {
            return null;
        }
        ListNode<T> head = new ListNode<>(elements[0]);
        for (int i = 1; i < elements.length; i++) {
            head.appendToTail(elements[i]);
        }
        return head;
    }

    // ex 1.1
    public static void removeDuplicates(ListNode<Integer> head) {
        if (head == null) {
            return;
        }

        HashSet<Integer> seenValues = new HashSet<Integer>();
        ListNode<Integer> previousNode = null;

        while (head != null) {
            if (seenValues.contains(head.data)) {
                previousNode.next = head.next;
            } else {
                seenValues.add(head.data);
                previousNode = head;
            }
            head = head.next;
        }
    }

    public static void main(String[] args) {
        ListNode<Integer> ex11answer = buildLinkedList(new Integer[]{1, 1, 1, 1, 1});
        removeDuplicates(ex11answer);
        ex11answer.print();
        ListNode<Integer> ex12answer = buildLinkedList(new Integer[]{1, 2, 1, 3, 1, 4, 0});
        removeDuplicates(ex12answer);
        ex12answer.print();
    }
}
