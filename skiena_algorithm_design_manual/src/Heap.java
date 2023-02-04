import java.util.ArrayList;
import java.util.Objects;

public class Heap<T extends Comparable<T>> {

    private ArrayList<T> arr;

    public Heap() {
        initializeArray();
    }

    public Heap(T[] list) {
        initializeArray();
        for (T elem : list) {
            insert(elem);
        }
    }

    private void initializeArray() {
        arr = new ArrayList<T>();
        arr.add(null);
    }

    public int size() {
        return arr.size() - 1;
    }

    public int parentIndex(int idx) {
        if (idx <= 1) {
            return -1;
        }
        return idx / 2; // implicitly does floor
    }

    public int youngChildIndex(int idx) {
        return 2 * idx;
    }

    public void insert(T elem) {
        arr.add(elem);
        bubbleUp(arr.size() - 1);
    }

    public T extractMin() {
        if (size() <= 0) {
            throw new java.lang.IllegalStateException("min value requested for empty priority queue!");
        }

        T min = arr.get(1);
        swap(1, size());
        arr.remove(size());
        bubbleDown(1);

        return min;
    }

    private void bubbleDown(int idx) {
        int leftChild = youngChildIndex(idx);
        int rightChild = leftChild + 1;
        int minIndex = idx;

        // Compare parent and left child.
        if (leftChild <= size() && arr.get(minIndex).compareTo(arr.get(leftChild)) > 0) {
            minIndex = leftChild;
        }
        // Compare the minimum with right child.
        if (rightChild <= size() && arr.get(minIndex).compareTo(arr.get(rightChild)) > 0) {
            minIndex = rightChild;
        }

        if (!Objects.equals(minIndex, idx)) {
            swap(minIndex, idx);
            bubbleDown(minIndex);
        }
    }

    private void bubbleUp(int idx) {
        if (parentIndex(idx) == -1) {
            return;
        }
        // parent should be larger than child in a heap
        if (arr.get(parentIndex(idx)).compareTo(arr.get(idx)) > 0) {
            swap(idx, parentIndex(idx));
            bubbleUp(parentIndex(idx));
        }
    }

    private void swap(int idx1, int idx2) {
        T temp = arr.get(idx1);
        arr.set(idx1, arr.get(idx2));
        arr.set(idx2, temp);
    }
}
