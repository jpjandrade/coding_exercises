import java.util.HashSet;

public class ArraysAndStrings {

    // ex 1.1
    static boolean isUnique(String str) {
        HashSet<Character> chars = new HashSet<Character>();
        for (Character c : str.toCharArray()) {
            chars.add(c);
        }
        return chars.size() == str.length();
    }

    // ex 1.3
    static void URLify(char[] inputString, int trueLength) {
        int numSpaces = 0, i = 0;
        for (i = 0; i < trueLength; i++) {
            if (inputString[i] == ' ') {
                numSpaces++;
            }
        }

        int idx = trueLength + 2 * numSpaces;
        for (i = trueLength - 1; i > 0; i--) {
            if (inputString[i] == ' ') {
                inputString[idx - 1] = '0';
                inputString[idx - 2] = '2';
                inputString[idx - 3] = '%';
                idx -= 3;
            } else {
                inputString[idx - 1] = inputString[i];
                idx--;
            }
        }
    }

    // ex 1.6
    static String compressString(String inputString) {
        if (inputString.length() == 0) {
            return inputString;
        }

        StringBuilder compressedStringBuilder = new StringBuilder();

        char currChar = inputString.charAt(0);
        int currCount = 1;

        for (char c : inputString.substring(1).toCharArray()) {
            if(c != currChar) {
                compressedStringBuilder.append(currChar);
                compressedStringBuilder.append(currCount);

                currChar = c;
                currCount = 0;
            }
            currCount++;
        }
        compressedStringBuilder.append(currChar);
        compressedStringBuilder.append(currCount);

        String compressedString = compressedStringBuilder.toString();
        if (compressedString.length() >= inputString.length()) {
            return inputString;
        }
        return compressedString;
    }

    public static void main(String[] args) {
        // ex 1.1 check
        System.out.printf("'abcde': %b\n", isUnique("abcde"));
        System.out.printf("'abcdea': %b\n", isUnique("abcdea"));

        // ex 1.6 check
        System.out.println(compressString("aabcccccaaa"));
        System.out.println(compressString("abcdefghij"));
        System.out.println(compressString("aaaaaaaaaaaaaaaaa"));
    }
}
