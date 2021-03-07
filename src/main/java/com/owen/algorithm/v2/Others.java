package com.owen.algorithm.v2;

import com.owen.algorithm.LinkList.ListNode;

import java.util.*;

public class Others {

    //[1].两数之和
    public static int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> hash = new HashMap<>();
        for (int index = 0; index < nums.length; index++) {

            if (hash.containsKey(target - nums[index])) {
                return new int[]{hash.get(target - nums[index]), index};
            }
            hash.put(nums[index], index);
        }
        return new int[0];
    }


    //[2]两数相加
    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummyHead = new ListNode(-1);
        ListNode cur = dummyHead;
        int carry = 0;
        while (l1 != null || l2 != null || carry != 0) {
            int a = 0;
            int b = 0;
            if (l1 != null) {
                a = l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                b = l2.val;
                l2 = l2.next;
            }
            int res = a + b + carry;

            cur.next = new ListNode(res % 10);
            carry = res / 10;
            cur = cur.next;
        }

        return dummyHead.next;
    }

    //[3]
    public static int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> window = new HashMap<>();
        char[] arr = s.toCharArray();
        int left = 0, maxLen = 0, right = 0;

        while (right < arr.length) {
            char r = arr[right];
            window.put(r, window.getOrDefault(r, 0) + 1);
            right++;

            while (window.get(r) > 1) {
                char l = arr[left];
                left++;
                window.put(l, window.get(l) - 1);
            }
            maxLen = Math.max(maxLen, right - left);
        }
        return maxLen;
    }

    //[7].整数反转
    public static int reverse(int x) {
        int res = 0;
        while (x != 0) {
            int a = x % 10;

            if (res > Integer.MAX_VALUE / 10 || (res == Integer.MAX_VALUE / 10 && a > 7)) return 0;
            if (res < Integer.MIN_VALUE / 10 || (res == Integer.MIN_VALUE / 10 && a < -8)) return 0;

            res = res * 10 + a;
            x /= 10;
        }
        return res;
    }

    //[8].字符串转换整数(atoi)
    public static int myAtoi(String s) {
        int sign = 1, res = 0;
        boolean skip = true;
        for (int i = 0; i < s.length(); i++) {
            if (skip && s.charAt(i) == ' ') {
                continue;
            } else if (skip && s.charAt(i) == '-') {
                sign = -1;
                skip = false;
            } else if (skip && s.charAt(i) == '+') {
                sign = 1;
                skip = false;
            } else if (s.charAt(i) >= '0' && s.charAt(i) <= '9') {
                int a = s.charAt(i) - '0';
                if (res > Integer.MAX_VALUE / 10 || (res == Integer.MAX_VALUE / 10 && a > Integer.MAX_VALUE % 10))
                    return Integer.MAX_VALUE;
                if (res < Integer.MIN_VALUE / 10 || (res == Integer.MIN_VALUE / 10 && a > -(Integer.MIN_VALUE % 10)))
                    return Integer.MIN_VALUE;

                res = res * 10 + sign * (s.charAt(i) - '0');
                skip = false;
            } else {
                break;
            }
        }
        return res;
    }

    //[9].回文数
    public static boolean isPalindrome(int x) {
        if (x < 0) return false;
        int cur = x, res = 0;
        while (cur != 0) {
            int b = cur % 10;
            res = res * 10 + b;
            cur /= 10;
        }
        return x == res;
    }

    //[12].整数转罗马数
    public static String intToRoman(int num) {
        //1994 ==> MCMXCIV
        int[] nums = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] chars = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};

        StringBuilder sb = new StringBuilder();
        for (int index = 0; index < nums.length; index++) {
            while (num >= nums[index]) {
                sb.append(chars[index]);
                num -= nums[index];
            }
        }
        return sb.toString();
    }

    //[13].罗马数转整数
    public static int romanToInt(String s) {
        int pre = getRomanNumber(s.charAt(0)), sum = 0;
        for (int i = 1; i < s.length(); i++) {
            int val = getRomanNumber(s.charAt(i));
            if (pre < val) {
                sum -= pre;
            } else {
                sum += pre;
            }
            pre = val;
        }
        sum += pre;
        return sum;
    }

    private static int getRomanNumber(char ch) {
        switch (ch) {
            case 'I':
                return 1;
            case 'V':
                return 5;
            case 'X':
                return 10;
            case 'L':
                return 50;
            case 'C':
                return 100;
            case 'D':
                return 500;
            case 'M':
                return 1000;
            default:
                return 0;
        }
    }

    //[14].最长公共前缀
    public static String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) return "";

        String res = strs[0];
        for (int i = 1; i < strs.length; i++) {
            res = longestCommonPrefix(res, strs[i]);
            if (res.length() == 0) {
                return "";
            }
        }
        return res;
    }

    private static String longestCommonPrefix(String first, String second) {
        int index = 0, minLen = Math.min(first.length(), second.length());
        while (index < minLen && (first.charAt(index) == second.charAt(index))) {
            index++;
        }
        return first.substring(0, index);
    }

    public static List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);

        List<List<Integer>> pos = new ArrayList<>();

        for (int i = 0; i < nums.length - 2; ) {
            int target = 0 - nums[i];
            List<List<Integer>> sums = twoSums(nums, i + 1, target);
            if (sums.size() > 0) {
                for (List<Integer> sum : sums) {
                    sum.add(nums[i]);
                    pos.add(sum);
                }
            }
            i++;
            while (i < nums.length - 2 && nums[i - 1] == nums[i]) {
                i++;
            }
        }
        return pos;
    }

    private static List<List<Integer>> twoSums(int[] nums, int start, int target) {
        List<List<Integer>> res = new ArrayList<>();
        int left = start, right = nums.length - 1;
        while (left < right) {
            int leftOne = nums[left];
            int rightOne = nums[right];
            int sum = leftOne + rightOne;
            if (sum == target) {
                List<Integer> group = new ArrayList<>();
                group.add(leftOne);
                group.add(rightOne);
                res.add(group);
                while (left < right && nums[left] == leftOne) left++;
                while (left < right && nums[right] == rightOne) right--;
            } else if (sum < target) {
                while (left < right && nums[left] == leftOne) left++;
            } else {
                while (left < right && nums[right] == rightOne) right--;
            }
        }
        return res;
    }

    //[16].最接近的三数之和
    public static int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int res = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.length - 2; ) {
            int num = nums[i];
            int left = i + 1, right = nums.length - 1;
            while (left < right) {
                int rightOne = nums[right], leftOne = nums[left];
                int sum = num + leftOne + rightOne;
                if (sum == target) {
                    return sum;
                }

                if (Math.abs(target - sum) < Math.abs(target - res)) {
                    res = sum;
                }
                if (sum > target) {
                    while (left < right && nums[right] == rightOne) right--;
                } else {
                    while (left < right && nums[left] == leftOne) left++;
                }
            }
            i++;
            while (i < nums.length - 2 && nums[i - 1] == nums[i]) i++;
        }
        return res;
    }


    //[17].电话号码的字母组合
    public static List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        if (digits.length() == 0) return res;
        String[] strings = new String[]{"-", "-", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        dfsForLetter(res, new StringBuilder(), digits, 0, strings);
        return res;
    }


    private static void dfsForLetter(List<String> res, StringBuilder select, String digits, int index, String[] strings) {
        if (digits.length() == select.length()) {
            res.add(select.toString());
            return;
        }

        String string = strings[digits.charAt(index) - '0'];
        for (int i = 0; i < string.length(); i++) {
            select.append(string.charAt(i));

            dfsForLetter(res, select, digits, index + 1, strings);
            select.deleteCharAt(select.length() - 1);
        }
    }


    public static void main(String[] args) {
        ListNode f = new ListNode(2);
        f.next = new ListNode(4);
        f.next.next = new ListNode(3);

        ListNode result = addTwoNumbers(f, null);
        while (result != null) {
            System.out.print("->" + result.val);
            result = result.next;
        }

        System.out.println(lengthOfLongestSubstring("abcabcdb"));
        System.out.println(lengthOfLongestSubstring(""));
        System.out.println(lengthOfLongestSubstring("pwwkew"));
        System.out.println(lengthOfLongestSubstring("abba"));

        System.out.println(reverse(Integer.MAX_VALUE));
        System.out.println(reverse(123));


        System.out.println(myAtoi("-2147483649"));
        System.out.println(myAtoi("-91283472332"));
        System.out.println(myAtoi("   -42"));
        System.out.println(myAtoi("9223372036854775808"));

        System.out.println(isPalindrome(2147447412));
        System.out.println(isPalindrome(123));

        System.out.println(intToRoman(1994));
        System.out.println(intToRoman(3999));

        System.out.println(romanToInt("MMMCMXCIX"));
        System.out.println(romanToInt("I"));


        System.out.println(longestCommonPrefix(new String[]{"dog", "racecar", "car"}));
        System.out.println(longestCommonPrefix(new String[]{"flower", "flow", "flight"}));


        System.out.println(threeSum(new int[]{-1, 0, 1, 2, -1, -4}));
        System.out.println(threeSum(new int[]{0, 0, 0, 0, 0, 0, 0}));
        System.out.println(threeSum(new int[]{-1, 0, 1, 2, -1, -4, -2, -3, 3, 0, 4}));

        System.out.println(threeSumClosest(new int[]{-1, 2, 1, -4}, 1));

        System.out.println(letterCombinations("23"));
        System.out.println(letterCombinations("22"));
    }
}
