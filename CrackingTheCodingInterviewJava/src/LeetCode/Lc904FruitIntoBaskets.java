package LeetCode;

import java.util.HashMap;

public class Lc904FruitIntoBaskets {
    public int totalFruit(int[] fruits) {
        HashMap<Integer, Integer> fruitCounts = new HashMap<Integer, Integer>();
        int b = 0;
        int longestChain = 0;

        for (int i = 0; i < fruits.length; i++) {
            fruitCounts.put(fruits[i], fruitCounts.getOrDefault(fruits[i], 0) + 1);
            while (fruitCounts.size() > 2) {
                longestChain = Integer.max(longestChain, i - b);
                fruitCounts.put(fruits[b], fruitCounts.get(fruits[b]) - 1);
                if (fruitCounts.get(fruits[b]) == 0) {
                    fruitCounts.remove(fruits[b]);
                }
                b++;
            }
        }
        longestChain = Integer.max(longestChain, fruits.length - b);
        return longestChain;
    }
}