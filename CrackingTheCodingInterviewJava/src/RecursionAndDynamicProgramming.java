import java.util.ArrayList;
import java.util.Arrays;

public class RecursionAndDynamicProgramming {

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

    public static int countWaysDP(int n) {
        if (n <= 0) {
            return 1;
        }
        int[] dp = new int[n];
        for (int i = 0; i < n; i++) {
            dp[i] = 0;
        }
        dp[0] = 1;
        if (n > 1) dp[1] = 2;
        if (n > 2) dp[2] = 2 + 1 + 1;
        for (int i = 3; i < n; i ++) {
            dp[i] = dp[i - 3] + dp[i - 2] + dp[i - 1];
        }
        return dp[n - 1];
    }


    // ex 8.2
//    public static char[] robotOnGrid() { return;}

    // ex 8.3
    int magicFind(int[] array) {
        return magicFind(array, 0, array.length - 1);
    }

    int magicFind(int[] array, int start, int end) {
        if (end < start) return -1;

        int midIndex = (start + end) / 2;
        int midValue = array[midIndex];
        if (midValue == midIndex) {
            return midIndex;
        }

        int leftIndex = Math.min(midIndex - 1, midValue);
        int left = magicFind(array, start, leftIndex);
        if (left > 0) {
            return left;
        }

        int rightIndex = Math.max(midIndex + 1, midValue);
        int right = magicFind(array, rightIndex, end);
        return right;
    }

    // ex 8.4

    private ArrayList<ArrayList<Integer>> getSubsets(ArrayList<Integer> set) {
        ArrayList<ArrayList<Integer>> allSubsets = new ArrayList<>();

        int max = 1 << set.size(); // 2^n
        for (int k = 0; k < max; k++) {
            ArrayList<Integer> subset = convertIntToSet(k, set);
            allSubsets.add(subset);
        }
        return allSubsets;
    }

    private ArrayList<Integer> convertIntToSet(int x, ArrayList<Integer> set) {
        ArrayList<Integer> subset = new ArrayList<>();
        int index = 0;
        for (int k = x; k > 0; k >>= 1) {
            if ((k & 1) == 1) {
                subset.add(set.get(index));
            }
            index++;
        }
        return subset;
    }
    // ex 8.12

    static int GRID_SIZE = 8;

    public static void placeQueens(int row, Integer[] columns, ArrayList<Integer[]> results) {
        if (row == GRID_SIZE) {
            results.add(columns.clone());
        } else {
            for (int col = 0; col < GRID_SIZE; col++) {
                if (checkValid(columns, row, col)) {
                    columns[row] = col;
                    placeQueens(row + 1, columns, results);
                }
            }
        }
    }

    private static boolean checkValid(Integer[] columns, int row1, int column1) {
        for (int row2 = 0; row2 < row1; row2++) {
            int column2 = columns[row2];

            if (column1 == column2) {
                return false;
            }

            int columnDistance = Math.abs(column2 - column1);

            int rowDistance = row1 - row2;
            if (columnDistance == rowDistance) {
                return false;
            }
        }
        return true;
    }


    public static void main(String[] args) {
        System.out.println("\nFibonacci example");
        for (int k=0; k < 10; k++) {
            System.out.print(fibonacci(k) + " ");
        }
        System.out.println("\nEx 8.1");
        for (int k=0; k < 10; k++) {
            System.out.print(countWays(k) + " ");
        }
        System.out.println();
        for (int k=0; k < 10; k++) {
            System.out.print(countWaysDP(k) + " ");
        }

        System.out.println("\nEx 8.12");
        Integer[] columns = new Integer[GRID_SIZE];
        ArrayList<Integer[]> results = new ArrayList<>();
        placeQueens(0, columns, results);
        System.out.println("\nQueen's positions");
        System.out.println(Arrays.toString(results.get(0)));

    }
}
