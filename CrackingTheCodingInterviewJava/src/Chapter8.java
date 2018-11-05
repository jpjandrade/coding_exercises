import java.util.Arrays;

public class Chapter8 {

    public static long fibonacci(int n) {
        return fibonacci(n, new long[n + 1]);
    }

    private static long fibonacci(int i, long[] memo) {
        if (i == 0 | i == 1) return i;

        if (memo[i] == 0) {
            memo[i] = fibonacci(i - 1, memo) + fibonacci(i - 2, memo);
        }
        return memo[i];

    }


    // ex 8.1
    private static int countWays(int n) {
        int[] memo = new int[n + 1];
        Arrays.fill(memo, -1);
        return countWays(n, memo);
    }

    public static int countWays(int n, int[] memo) {
        if (n < 0) {
            return 0;
        }
        else if (n == 0) {
            return 1;
        }
        else if (memo[n] > -1) {
            return memo[n];
        }
        else {
            memo[n] = countWays(n - 1, memo) + countWays(n - 2, memo) + countWays(n - 3, memo);
            return memo[n];
        }
    }

    public static void main(String[] args) {
        System.out.println("\nFibonacci example");
        for (int k=0; k < 10; k++) {
            System.out.print(fibonacci(k) + " ");
        }
        System.out.println("\nEx 4.1");
        for (int k=0; k < 10; k++) {
            System.out.print(countWays(k) + " ");
        }
    }
}
