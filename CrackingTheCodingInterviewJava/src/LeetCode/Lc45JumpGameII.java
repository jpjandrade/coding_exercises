package LeetCode;

public class Lc45JumpGameII {
    // this turned out to be O(n^2), not got enough :(
    public int jump(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = 0;
        for (int i = 1; i < nums.length; i++) {
            int minJumps = nums.length + 1;
            for (int k = 0; k < i; k++) {
                if (nums[k] + k >= i) {
                    minJumps = Integer.min(minJumps, dp[k]);
                }
            }
            dp[i] = minJumps + 1;

        }
        return dp[nums.length - 1];
    }
}
