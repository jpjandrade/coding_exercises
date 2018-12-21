import java.util.Random;


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
}
