package com.owen.algorithm.v3;

public class Test {

    public static void main(String[] args) {
        System.out.println(new Test().reverseWords("the sky is blue"));
    }

    public String reverseWords(String s) {
        StringBuilder sb = trim(s);
        reverse(sb,0, sb.length() - 1);
        reverseEachWord(sb);
        return sb.toString();
    }

    private void reverse(StringBuilder sb, int left, int right) {
        while (left < right) {
            char tmp = sb.charAt(left);
            sb.setCharAt(left++, sb.charAt(right));
            sb.setCharAt(right--, tmp);
        }
    }

    private void reverseEachWord(StringBuilder sb) {
        int i = 0;
        while (i < sb.length()) {
            int j = i;
            while (j < sb.length() && sb.charAt(j) != ' ') {
                j++;
            }
            reverse(sb, i, j - 1);
            i = j + 1;
        }
    }

    private StringBuilder trim(String s) {
        StringBuilder sb = new StringBuilder();
        int left = 0, right = s.length() - 1;
        while (left <= right && s.charAt(left) == ' ') left++;
        while (left <= right && s.charAt(right) == ' ') right--;

        while (left <= right) {
            char c = s.charAt(left++);
            if (c != ' ') {
                sb.append(c);
            } else if (sb.charAt(sb.length() - 1) != ' ') {
                sb.append(c);
            }
        }
        return sb;
    }
}
