import java.util.LinkedList;

public class Sorting {
    public static <T extends Comparable<T>> void heapSort(T[] arr) {
        Heap<T> h = new Heap<T>(arr);
        for (int i = 0; i < arr.length; i++) {
            arr[i] = h.extractMin();
        }
    }

    public static <T extends Comparable<T>> void mergeSort(T[] arr) {
        mergeSort(arr, 0, arr.length - 1);
    }

    private static <T extends Comparable<T>> void mergeSort(T[] arr, int low, int high) {
        if (low >= high) {
            return;
        }
        int middle = (low + high) / 2;
        mergeSort(arr, low, middle);
        mergeSort(arr, middle + 1, high);
        merge(arr, low, middle, high);
    }

    private static <T extends Comparable<T>> void merge(T[] arr, int low, int middle, int high) {
        LinkedList<T> buffer1 = new LinkedList<T>();
        LinkedList<T> buffer2 = new LinkedList<T>();

        for(int i = low; i<=middle; i++) buffer1.add(arr[i]);
        for(int i = middle + 1; i<= high; i++) buffer2.add(arr[i]);

        int i = low;
        while (!(buffer1.isEmpty() || buffer2.isEmpty())) {
            if (buffer1.peek().compareTo(buffer2.peek()) <= 0) {
                arr[i] = buffer1.pop();
            } else{
                arr[i] = buffer2.pop();
            }
            i++;
        }

        while(!buffer1.isEmpty()) {
            arr[i] = buffer1.pop();
            i++;
        }

        while(!buffer2.isEmpty()) {
            arr[i] = buffer2.pop();
            i++;
        }

    }

    public static <T extends Comparable<T>> void insertionSort(T[] arr) {
        int min;
        for (int i = 0; i < arr.length; i++) {
            min = i;
            for (int j = i + 1; j < arr.length; j++) {
                if (arr[j].compareTo(arr[min]) < 0) min = j;
            }
            swap(arr, i, min);
        }
    }

    private static <T> void swap(T[] arr, int idx1, int idx2) {
        T temp = arr[idx1];
        arr[idx1] = arr[idx2];
        arr[idx2] = temp;
    }

}
