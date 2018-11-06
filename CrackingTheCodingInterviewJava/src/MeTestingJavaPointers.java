import java.util.Arrays;

public class MeTestingJavaPointers {

    private static void foo(int[] arr) {
        for(int k=0; k < arr.length; k++) {
            if(k % 2 == 0) {
                arr[k] = k;
            }
        }
    }
    public static void main(String[] args) {
        int[] a = new int[10];
        System.out.println(Arrays.toString(a));
        foo(a);
        System.out.println(Arrays.toString(a));
    }
}
