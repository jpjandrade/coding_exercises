import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        Integer[] arr = new Integer[]{3, 1, 4, 4, 10, 1, -1, 4};

        Integer[] heapArr = Arrays.copyOf(arr, arr.length);
        Sorting.heapSort(heapArr);
        System.out.println(Arrays.toString(heapArr));

        Integer[] insertArr = Arrays.copyOf(arr, arr.length);
        Sorting.insertionSort(insertArr);
        System.out.println(Arrays.toString(insertArr));
    }
}