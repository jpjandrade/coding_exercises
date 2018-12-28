import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class MeTestingJava {

    // is java pass by value or pass by reference (ans: pass by reference, arr is modified here)
    private static void foo(int[] arr) {
        for(int k=0; k < arr.length; k++) {
            if(k % 2 == 0) {
                arr[k] = k;
            }
        }
    }

    // quickselect attempt
    public static int quickselect(int[] arr, int k) {
        return quickselect(arr, 0, arr.length - 1, k);
    }

    private static int quickselect(int[] arr, int left, int right, int k) {
        if (left == right) {
            return arr[left];
        }
        int index = SortingAndSearching.partition(arr, left, right);
        if (index == k) {
            return arr[k];
        }
        if (left < index - 1) {
            return quickselect(arr, left, index - 1, k);
        }

        return quickselect(arr, index, right, k);

    }



    // leetcode exercise
    public static int findMaximumXOR(int[] nums) {
        int max = 0, mask = 0;
        for(int i = 31; i >= 0; i--){
            mask = mask | (1 << i);
            System.out.println(Integer.toBinaryString(mask));
            Set<Integer> set = new HashSet<>();
            for(int num : nums){
                set.add(num & mask);
            }
            System.out.println(set.toString());
            int tmp = max | (1 << i);
            for(int prefix : set){
                if(set.contains(tmp ^ prefix)) {
                    System.out.print("New max! ");
                    System.out.println(Integer.toBinaryString(tmp));
                    max = tmp;
                    break;
                }
            }
        }
        return max;
    }

    public static void main(String[] args) {
        int[] a = new int[10];
        System.out.println(Arrays.toString(a));
        foo(a);
        System.out.println(Arrays.toString(a));

        // int[] nums = new int[]{2, 3, 6, 10, 25};
        // int r = findMaximumXOR(nums);
        // System.out.println(r);
        int [] arr = new int[] {0, -1, 4, 2, 10, 5};
        System.out.println(quickselect(arr, 0));
    }
}
