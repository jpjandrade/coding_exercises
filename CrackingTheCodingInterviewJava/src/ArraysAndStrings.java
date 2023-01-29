import java.util.HashSet;
public class ArraysAndStrings {

    // ex 1.1
    static boolean isUnique(String str) {
        HashSet<Character> chars = new HashSet<Character>();
        for (char c : str.toCharArray()) {
            chars.add(c);
        }
        return chars.size() == str.length();
    }

    public static void main(String[] args) {
        System.out.printf("'abcde': %b\n", isUnique("abcde"));
        System.out.printf("'abcdea': %b\n", isUnique("abcdea"));
    }
}
