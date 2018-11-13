import java.util.Arrays;

public class SortingAndSearching {

    // mergesort
    public static void mergesort(int[] array) {
        int[] helper = new int[array.length];
        mergesort(array, helper, 0, array.length - 1);
    }

    private static void mergesort(int[] array, int[] helper, int low, int high) {
        if (low < high) {
            int middle = (low + high) / 2;
            mergesort(array, helper, low, middle);
            mergesort(array, helper, middle + 1, high);
            merge(array, helper, low, middle, high);
        }
    }

    private static void merge(int[] array, int[] helper, int low, int middle, int high) {
        for(int i = low; i <= high; i++) {
            helper[i] = array[i];
        }
        int helperLeft = low;
        int helperRight = middle + 1;
        int current = low;

        while(helperLeft <= middle && helperRight <= high) {
            if (helper[helperLeft] <= helper[helperRight]) {
                array[current] = helper[helperLeft];
                helperLeft++;
            }
            else {
                array[current] = helper[helperRight];
                helperRight++;
            }
            current++;
        }

        int remaining = middle - helperLeft;
        for (int i = 0; i <= remaining; i++) {
            array[current + i] = helper[helperLeft + i];
        }
    }

    // quicksort
    public static void quicksort(int[] arr) {
        quicksort(arr, 0, arr.length - 1);
    }

    private static void quicksort(int[] arr, int left, int right) {
        int index = partition(arr, left, right);
        if (left < index - 1) {
            quicksort(arr, left, index - 1);
        }
        if (index < right) {
            quicksort(arr, index, right);
        }
    }

    private static int partition(int[] arr, int left, int right) {
        int pivot = arr[(left + right) / 2];
        while (left <= right) {
            while (arr[left] < pivot) left++;

            while (arr[right] > pivot) right--;

            if (left <= right) {
                swap(arr, left, right);
                left++;
                right--;
            }
        }
        return left;
    }

    private static void swap(int[] arr, int left, int right) {
        int temp = arr[left];
        arr[left] = arr[right];
        arr[right] = temp;
    }


    // binary search

    public static int binarySearch(int[] a, int x) {
        int low = 0;
        int high = a.length - 1;
        int mid;

        while (low <= high) {
            mid = (low + high) / 2;
            if (a[mid] < x) {
                low = mid + 1;
            } else if (a[mid] > x){
                high = mid - 1;
            } else {
                return mid;
            }

        }
        return -1;
    }

    public static int binarySearchRecursive(int[] a, int x, int low, int high) {
        if (low > high) return -1;

        int mid = (low + high) / 2;
        if (a[mid] > x) {
            return binarySearchRecursive(a, x, mid + 1, high);
        }
        else if (a[mid] < x) {
            return binarySearchRecursive(a, x, low, mid - 1);
        }
        else {
            return mid;
        }
    }


    public static void main(String[] args){
        int[] arr = new int[]{0, 4, 2, 3, 1, -2, 10, 3};
        System.out.println(Arrays.toString(arr));
        mergesort(arr);
        System.out.println(Arrays.toString(arr));

        int[] arr2 = new int[]{0, 4, 2, 3, 1, -2, 10, 3};
        quicksort(arr2);
        System.out.println(Arrays.toString(arr));

        int indexOf4 = binarySearch(arr2, 4);
        System.out.println(indexOf4);



    }



}
