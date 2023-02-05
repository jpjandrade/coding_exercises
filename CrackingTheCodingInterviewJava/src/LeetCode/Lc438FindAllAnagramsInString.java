package LeetCode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Lc438FindAllAnagramsInString {
    public List<Integer> findAnagrams(String s, String p) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        int windowSize = p.length();

        HashMap<Character, Integer> pChars = new HashMap<Character, Integer>();
        HashMap<Character, Integer> sChars = new HashMap<Character, Integer>();

        for (char c : p.toCharArray()) {
            pChars.put(c, pChars.getOrDefault(c, 0) + 1);
        }

        for (int i = 0; i < s.length(); i++) {
            sChars.put(s.charAt(i), sChars.getOrDefault(s.charAt(i), 0) + 1);
            if (i >= windowSize) {
                sChars.put(s.charAt(i - windowSize), sChars.get(s.charAt(i - windowSize)) - 1);

                if (sChars.get(s.charAt(i - windowSize)) == 0) {
                    sChars.remove(s.charAt(i - windowSize));
                }
            }
            if (pChars.equals(sChars)) {
                result.add(i + 1 - windowSize);
            }
        }

        return result.stream().toList();
    }
}
