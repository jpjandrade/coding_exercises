import java.lang.reflect.Array;
import java.util.*;


public class HardProblems {
    // ex 17.1
    public int addNoSum(int a, int b) {
        if (b == 0) return a;
        int sum = a ^ b;
        int carry = (a & b) << 1;
        return addNoSum(sum, carry);
    }

    // ex 17.2
    public void shuffleArray(int[] cards) {
        for (int i = 0; i < cards.length; i++) {
            int k = new Random().nextInt(i + 1); // i inclusive
            int temp = cards[k];
            cards[k] = cards[i];
            cards[i] = temp;
        }
    }

    public int[] pickM(int[] original, int m) {
        int[] subset = new int[m];
        for (int i = 0; i < m; i++){
            subset[i] = original[i];
        }

        for (int i = m; i < original.length; i++) {
            int k = new Random().nextInt(i + 1);
            if (k < m) {
                subset[k] = original[i];
            }
        }

        return subset;
    }

    // ex 17.6

    int count2sInRangeAtDigit(int number, int d) {
        int powerOf10 = (int) Math.pow(10, d);
        int nextPowerOf10 = powerOf10 * 10;
        int right = number % powerOf10;

        int roundDown = number - number % nextPowerOf10;
        int roundUp = roundDown + nextPowerOf10;

        int digit = (number / powerOf10) % 10;
        if (digit < 2) {
            return roundDown / 10;
        } else if (digit == 2){
            return roundDown / 10 + right + 1;
        } else {
            return roundUp / 10;
        }
    }

    int count2sInRange(int number) {
        int count = 0;
        int len = String.valueOf(number).length();
        for (int digit = 0; digit < len; digit++) {
            count += count2sInRangeAtDigit(number, digit);
        }
        return count;
    }

    // ex 17.9

    int getKthMagicNumber(int k) {
        if (k < 0) {
            return 0;
        }
        int val = 0;
        Queue<Integer> queue3 = new Queue<>();
        Queue<Integer> queue5 = new Queue<>();
        Queue<Integer> queue7 = new Queue<>();
        queue3.add(1);

        for (int i = 0; i <= k; i++) {
            int v3 = queue3.isEmpty() ? Integer.MAX_VALUE : queue3.peek();
            int v5 = queue5.isEmpty() ? Integer.MAX_VALUE : queue5.peek();
            int v7 = queue7.isEmpty() ? Integer.MAX_VALUE : queue7.peek();
            val = Math.min(v3, Math.min(v5, v7));
            if (val == v3) {
                queue3.remove();
                queue3.add(3 * val);
                queue5.add(5 * val);
            } else if (val == v5) {
                queue5.remove();
                queue5.add(5 * val);
            } else if (val == v7) {
                queue7.remove();
            }
            queue7.add(7 * val);
        }
        return val;
    }
    // ex 17.21

    int computeHistogramVolume(int[] histo) {
        int[] leftMaxes = new int[histo.length];
        int leftMax = histo[0];
        for (int i = 0; i < histo.length; i++) {
            leftMax = Math.max(leftMax, histo[i]);
            leftMaxes[i] = leftMax;
        }

        int sum = 0;

        int rightMax = histo[histo.length - 1];
        for (int i = histo.length - 1; i >= 0; i--) {
            rightMax = Math.max(rightMax, histo[i]);
            int secondTallest = Math.min(rightMax, leftMaxes[i]);

            if  (secondTallest > histo[i]) {
                sum += secondTallest - histo[i];
            }
        }

        return sum;
    }

    // leetcode 4, from https://www.youtube.com/watch?v=LPFhl65R7ww
    double medianOfSortedArrays(int[] arr1, int[] arr2) {
        if (arr1.length > arr2.length) {
            return medianOfSortedArrays(arr2, arr1);
        }

        int n = arr1.length;
        int m = arr2.length;

        int low = 0;
        int high = n;

        while (low <= high) {
            int i = (low + high) / 2;
            int j = (n + m + 1) / 2 - i;

            int maxLeft1 = (i == 0) ? Integer.MIN_VALUE : arr1[i - 1];
            int minRight1 = (i == n) ? Integer.MAX_VALUE : arr1[i];

            int maxLeft2 = (j == 0) ? Integer.MIN_VALUE : arr2[j - 1];
            int minRight2 = (j == m) ? Integer.MAX_VALUE : arr2[j];

            if (maxLeft1 <= minRight2 && maxLeft2 <= minRight1) {
                if ((n + m) % 2 == 0){
                    double median = (Math.max(maxLeft1, maxLeft2) + Math.min(minRight1, minRight2)) / 2.0;
                    return median;
                } else {
                    return (double)Math.max(maxLeft1, maxLeft2);
                }

            } else if (maxLeft1 > minRight2) {
                high = i - 1;
            } else {
                low = i + 1;
            }
        }

        throw new IllegalArgumentException("Lists not ordered!");
    }
}
