import java.util.HashMap;

// ex 16.25
public class LRUCache {
    private int maxCacheSize;
    private HashMap<Integer, LinkedListNode> map = new HashMap<Integer, LinkedListNode>();
    private LinkedListNode head = null;
    public LinkedListNode tail = null;

    public LRUCache(int maxSize) {
        maxCacheSize = maxSize;
    }

    public String getValue(int key) {
        LinkedListNode item = map.get(key);
        if (item == null) return null;

        if (item != head) {
            removeFromLinkedList(item);
            insertAtFrontOfList(item);
        }


        return item.value;
    }

    private void removeFromLinkedList(LinkedListNode node) {
        if (node == null) return;
        if (node.prev != null) node.prev.next = node.next;
        if (node.next != null) node.next.prev = node.prev;
        if (node == tail) tail = node.prev;
        if (node == head) head = node.next;
    }

    private void insertAtFrontOfList(LinkedListNode node) {
        if (head == null) {
            head = node;
            tail = node;
        } else {
            head.prev = node.next;
            node.next = head;
            head = node;
        }
    }

    public boolean removeKey(int key) {
        LinkedListNode node = map.get(key);
        removeFromLinkedList(node);
        map.remove(key);
        return true;
    }

    public void setKeyValue(int key, String value) {
        removeKey(key);

        if (map.size() >= maxCacheSize && tail != null) {
            removeKey(tail.key);
        }

        LinkedListNode node = new LinkedListNode(key, value);
        insertAtFrontOfList(node);
        map.put(key, node);
    }

    private static class LinkedListNode {
        private LinkedListNode next, prev;
        public int key;
        public String value;
        public LinkedListNode(int k, String v) {
            key = k;
            value = v;
        }
    }
}

