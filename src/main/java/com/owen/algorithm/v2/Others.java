package com.owen.algorithm.v2;

import com.owen.algorithm.LinkList;
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

    //[19].
    public static ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;
        ListNode slow = dummyHead, fast = dummyHead;
        while (n-- > 0) {
            fast = fast.next;
        }

        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return dummyHead.next;
    }

    //[20].有效括号
    public static boolean isValid(String s) {
        if (s == null || s.length() == 0) return true;
        Stack<Character> stack = new Stack<>();
        char[] chars = s.toCharArray();
        for (char ch : chars) {
            if (ch == '{' || ch == '(' || ch == '[') {
                stack.push(ch);
            } else if (ch == '}') {
                if (stack.isEmpty() || stack.peek() != '{') return false;
                else stack.pop();
            } else if (ch == ']') {
                if (stack.isEmpty() || stack.peek() != '[') return false;
                else stack.pop();
            } else if (ch == ')') {
                if (stack.isEmpty() || stack.peek() != '(') return false;
                else stack.pop();
            }
        }
        return stack.isEmpty();
    }

    //[21].合并两个有序链表
    public static ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummyHead = new ListNode(0), cur = dummyHead;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                cur.next = l1;
                l1 = l1.next;
                cur = cur.next;
            } else {
                cur.next = l2;
                l2 = l2.next;
                cur = cur.next;
            }
        }
        cur.next = l1 == null ? l2 : l1;
        return dummyHead.next;
    }

    //[22].括号生成
    public static List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        if (n < 0) return res;
        dfsForGenerateParenthesis(n, n, new StringBuilder(), res);
        return res;
    }

    private static void dfsForGenerateParenthesis(int left, int right, StringBuilder sb, List<String> res) {
        if (left > right || left < 0 || right < 0) {
            return;
        }

        if (left == 0 && right == 0) {
            res.add(sb.toString());
            return;
        }
        sb.append('(');
        dfsForGenerateParenthesis(left - 1, right, sb, res);
        sb.deleteCharAt(sb.length() - 1);

        sb.append(')');
        dfsForGenerateParenthesis(left, right - 1, sb, res);
        sb.deleteCharAt(sb.length() - 1);
    }

    //[23].合并k个升序链表
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> queue = new PriorityQueue<>(Comparator.comparingInt(o -> o.val));
        for (int i = 0; i < lists.length; i++) {
            if (lists[i] != null) {
                queue.offer(lists[i]);
            }
        }

        ListNode dummyHead = new ListNode(-1), cur = dummyHead;
        while (!queue.isEmpty()) {
            ListNode top = queue.poll();
            cur.next = top;
            cur = cur.next;
            if (top.next != null) {
                queue.offer(top.next);
            }
        }
        return dummyHead.next;
    }

    //[24].两两交换链表中的节点
    public static ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode next = head.next;
        ListNode newHead = swapPairs(next.next);
        next.next = head;
        head.next = newHead;
        return next;
    }

    public static ListNode reverseKGroup(ListNode head, int k) {
        ListNode start = head, end = head;
        for (int i = 0; i < k; i++) {
            if (end == null) return head;
            end = end.next;
        }

        ListNode dummyHead = new ListNode(-1);
        ListNode cur = start;
        while (cur != end) {
            ListNode dummyHeadNext = dummyHead.next;
            ListNode next = cur.next;
            cur.next = dummyHeadNext;
            dummyHead.next = cur;
            cur = next;
        }

        start.next = reverseKGroup(end, k);
        return dummyHead.next;
    }

    //[26].刪除排序数组中的重复项
    public static int removeDuplicates(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return 1;
        int slow = 0, fast = 1;
        while (fast < nums.length) {
            if (nums[slow] == nums[fast]) {
                fast++;
            } else {
                nums[++slow] = nums[fast];
                fast++;
            }
        }
        return slow + 1;
    }

    //[27].移除元素
    public static int removeElement(int[] nums, int val) {
        int slow = -1, fast = 0;
        while (fast < nums.length) {
            if (nums[fast] == val) {
                fast++;
            } else {
                nums[++slow] = nums[fast];
                fast++;
            }
        }
        return slow + 1;
    }

    //[31].下一个排列
    public static void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            i--;
        }

        if (i >= 0) {
            int j = nums.length - 1;
            while (j >= 0 && nums[i] >= nums[j]) {
                j--;
            }
            swap(nums, i, j);
        }
        reverse(nums, i + 1);
    }

    private static void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    private static void reverse(int[] nums, int start) {
        int end = nums.length - 1;
        while (start < end) {
            swap(nums, start, end);
            end--;
            start++;
        }
    }

    //[33].搜索旋转排序数组
    public static int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            }

            if (nums[mid] > nums[right]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }

    //[34].再排序数组中查找元素的第一个和最后一个位置
    public static int[] searchRange(int[] nums, int target) {
        int left = findIndex(nums, target, true);
        int right = findIndex(nums, target, false);
        return new int[]{left, right};
    }

    private static int findIndex(int[] nums, int target, boolean isLeft) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                if (isLeft) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (isLeft && (left >= nums.length || nums[left] != target)) return -1;
        if (!isLeft && (right < 0 || nums[right] != target)) return -1;
        return isLeft ? left : right;
    }

    //[35].搜索插入位置
    public static int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (target == nums[mid]) {
                return mid;
            } else if (target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    //[36].有效地数独
    public static boolean isValidSudoku(char[][] board) {
        for (int i = 0; i < 9; i++) {
            int[] rowCount = new int[10];
            int[] colCount = new int[10];
            int[] areaCount = new int[10];
            for (int j = 0; j < 9; j++) {
                char ch = board[i][j];
                if (ch != '.') {
                    if (rowCount[ch - '0'] > 0) {
                        return false;
                    } else {
                        rowCount[ch - '0']++;
                    }
                }

                ch = board[j][i];
                if (ch != '.') {
                    if (colCount[ch - '0'] > 0) {
                        return false;
                    } else {
                        colCount[ch - '0']++;
                    }
                }

                int x = (i / 3) * 3 + j / 3;
                int y = (i % 3) * 3 + j % 3;
                ch = board[x][y];
                if (ch != '.') {
                    if (areaCount[ch - '0'] > 0) {
                        return false;
                    } else {
                        areaCount[ch - '0']++;
                    }
                }
            }
        }
        return true;
    }

    //[38].外观数列
    public static String countAndSay(int n) {
        if (n == 1) {
            return "1";
        }

        String s = countAndSay(n - 1);
        StringBuilder sb = new StringBuilder();
        int start = 0;
        for (int right = 1; right < s.length(); right++) {
            if (s.charAt(start) != s.charAt(right)) {
                sb.append(right - start).append(s.charAt(start));
                start = right;
            }
        }
        sb.append(s.length() - start).append(s.charAt(start));
        return sb.toString();
    }

    //[39].组合总和
    public static List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        dfsForCombinationSum(candidates, 0, target, new LinkedList<>(), res);
        return res;
    }

    private static void dfsForCombinationSum(int[] candidates, int start, int target, LinkedList<Integer> select, List<List<Integer>> res) {
        if (target == 0) {
            res.add(new ArrayList<>(select));
            return;
        }
        //用start变量代表只能往后选择，不能往前选择
        for (int i = start; i < candidates.length; i++) {
            int candidate = candidates[i];
            if (target >= candidate) {
                select.addLast(candidate);
                dfsForCombinationSum(candidates, i, target - candidate, select, res);
                select.removeLast();
            }
        }
    }

    //[40].组合总和II
    public static List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(candidates);
        dfsForCombinationSum2(candidates, 0, target, new LinkedList<>(), res);
        return res;
    }

    private static void dfsForCombinationSum2(int[] candidates, int start, int target, LinkedList<Integer> select, List<List<Integer>> res) {
        if (target == 0) {
            res.add(new ArrayList<>(select));
            return;
        }

        //用start变量代表只能往后选择，不能往前选择
        //本层选择中，跳过重复的数字，解集不能包含重复的组合。
        for (int i = start; i < candidates.length; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
            int candidate = candidates[i];
            if (target >= candidate) {
                select.addLast(candidate);
                dfsForCombinationSum2(candidates, i + 1, target - candidate, select, res);
                select.removeLast();
            }
        }

    }

    //[45].跳跃游戏II
    public static int jump(int[] nums) {
        int end = 0, steps = 0, maxPos = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            maxPos = Math.max(maxPos, i + nums[i]);
            if (end == i) {
                end = maxPos;
                steps++;
            }
        }
        return steps;
    }

    //[46].全排列
    public static List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        dfsForPremute(nums, new LinkedList<>(), res);
        return res;
    }

    private static void dfsForPremute(int[] nums, LinkedList<Integer> select, List<List<Integer>> res) {
        if (select.size() == nums.length) {
            res.add(new ArrayList<>(select));
            return;
        }

        //没有重复的数字
        for (int num : nums) {
            if (!select.contains(num)) {
                select.addLast(num);
                dfsForPremute(nums, select, res);
                select.removeLast();
            }
        }
    }

    public static List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        dfsForPermuteUnique(nums, new boolean[nums.length], new LinkedList<>(), res);
        return res;
    }

    private static void dfsForPermuteUnique(int[] nums, boolean[] visited, LinkedList<Integer> select, List<List<Integer>> res) {
        if (nums.length == select.size()) {
            res.add(new ArrayList<>(select));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            //本层选择中如果前一个被选择过，并且相同，则跳过。
            if (i > 0 && nums[i] == nums[i - 1]) {
                if (visited[i - 1]) {
                    continue;
                }
            }

            //递归选择中当前被选择过，跳过。
            if (visited[i]) {
                continue;
            }

            select.addLast(nums[i]);
            visited[i] = true;
            dfsForPermuteUnique(nums, visited, select, res);
            select.removeLast();
            visited[i] = false;
        }
    }


    //[48].旋转图像
    public static void rotate(int[][] matrix) {
        int n = matrix.length;
        if (n == 1) {
            return;
        }
        //外向内共n/2层需要旋转
        for (int i = 0; i < n / 2; i++) {
            //每一层需要处理的列数
            for (int j = i; j < n - i - 1; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - j][i];
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j];
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i];
                matrix[j][n - 1 - i] = tmp;
            }
        }
    }


    //[50].Pow(x,n)
    public static double myPow(double x, int n) {
        if (n == 0) return 1;
        if (n == 1) return x;
        if (n == -1) return 1 / x;

        double half = myPow(x, n / 2);
        double rest = myPow(x, n % 2);
        return half * half * rest;
    }

    //[53].最大子序和
    public static int maxSubArray(int[] nums) {
        //sum表示以nums[i]为结尾的最大子序列和，前面为负数，则取当前值。否则扩充子序列和。
        int sum = nums[0], ans = 0;
        for (int i = 1; i < nums.length; i++) {
            sum = Math.max(sum + nums[i], nums[i]);
            ans = Math.max(sum, ans);
        }
        return ans;
    }

    //[55].跳跃游戏
    public static boolean canJump(int[] nums) {
        int end = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            end = Math.max(end, i + nums[i]);
            if (i >= end) {
                return false;
            }
        }
        return true;
    }

    public static void main(String[] args) {
//        ListNode f = new ListNode(2);
//        f.next = new ListNode(4);
//        f.next.next = new ListNode(3);
//
//        ListNode result = addTwoNumbers(f, null);
//        while (result != null) {
//            System.out.print("->" + result.val);
//            result = result.next;
//        }
//
//        System.out.println(lengthOfLongestSubstring("abcabcdb"));
//        System.out.println(lengthOfLongestSubstring(""));
//        System.out.println(lengthOfLongestSubstring("pwwkew"));
//        System.out.println(lengthOfLongestSubstring("abba"));
//
//        System.out.println(reverse(Integer.MAX_VALUE));
//        System.out.println(reverse(123));
//
//
//        System.out.println(myAtoi("-2147483649"));
//        System.out.println(myAtoi("-91283472332"));
//        System.out.println(myAtoi("   -42"));
//        System.out.println(myAtoi("9223372036854775808"));
//
//        System.out.println(isPalindrome(2147447412));
//        System.out.println(isPalindrome(123));
//
//        System.out.println(intToRoman(1994));
//        System.out.println(intToRoman(3999));
//
//        System.out.println(romanToInt("MMMCMXCIX"));
//        System.out.println(romanToInt("I"));
//
//
//        System.out.println(longestCommonPrefix(new String[]{"dog", "racecar", "car"}));
//        System.out.println(longestCommonPrefix(new String[]{"flower", "flow", "flight"}));
//
//
//        System.out.println(threeSum(new int[]{-1, 0, 1, 2, -1, -4}));
//        System.out.println(threeSum(new int[]{0, 0, 0, 0, 0, 0, 0}));
//        System.out.println(threeSum(new int[]{-1, 0, 1, 2, -1, -4, -2, -3, 3, 0, 4}));
//
//        System.out.println(threeSumClosest(new int[]{-1, 2, 1, -4}, 1));
//
//        System.out.println(letterCombinations("23"));
//        System.out.println(letterCombinations("22"));
//
//        ListNode l1 = new ListNode(1);
//        l1.next = new ListNode(2);
//        ListNode res = removeNthFromEnd(l1, 1);
//        System.out.println();
//
//        System.out.println(isValid("{[]}"));
//
//        System.out.println(removeDuplicates(new int[]{0, 0, 0, 1, 1, 1}));
//        System.out.println(removeDuplicates(new int[]{0, 0, 1, 1, 1, 2, 2, 3, 3, 4}));
//        System.out.println(removeDuplicates(new int[]{1, 1, 2}));
//
//        System.out.println(removeElement(new int[]{2, 2, 3, 4, 5}, 2));
//        System.out.println(removeElement(new int[]{0, 1, 2, 2, 3, 0, 4, 2}, 2));


//        System.out.println(generateParenthesis(3));

//        ListNode f = new ListNode(1);
//        f.next = new ListNode(2);
//        f.next.next = new ListNode(3);
//        f.next.next.next = new ListNode(4);
//        f.next.next.next.next = new ListNode(5);
//        f.next.next.next.next.next = new ListNode(6);
//        f.next.next.next.next.next.next = new ListNode(7);
//        f.next.next.next.next.next.next.next = new ListNode(8);
//        f.next.next.next.next.next.next.next.next = new ListNode(9);
//
//        ListNode res = reverseKGroup(f, 3);
//        System.out.println();

//        int[] arr = new int[]{3,2,1};
//        nextPermutation(arr);
//        System.out.println(Arrays.toString(arr));
//
//        arr = new int[]{4,5,2,6,3,1};
//        nextPermutation(arr);
//        System.out.println(Arrays.toString(arr));

//        arr = new int[]{1,1};
//        nextPermutation(arr);
//        System.out.println(Arrays.toString(arr));

//        System.out.println(searchInsert(new int[]{1,3,5,6}, 5));
//        System.out.println(searchInsert(new int[]{1,3,5,6}, 2));
//        System.out.println(searchInsert(new int[]{1,3,5,6}, 7));
//        System.out.println(searchInsert(new int[]{1,3,5,6}, 0));

//        System.out.println(isValidSudoku(new char[][]{{'8','3','.','.','7','.','.','.','.'},
//                {'6','.','.','1','9','5','.','.','.'},
//                {'.','9','8','.','.','.','.','6','.'},
//                {'8','.','.','.','6','.','.','.','3'},
//                {'4','.','.','8','.','3','.','.','1'},
//                {'7','.','.','.','2','.','.','.','6'},
//                {'.','6','.','.','.','.','2','8','.'},
//                {'.','.','.','4','1','9','.','.','5'},
//                {'.','.','.','.','8','.','.','7','9'}}));

//        System.out.println(countAndSay(7));

//        System.out.println(combinationSum(new int[]{2, 3, 6, 7}, 7));
//        System.out.println(combinationSum(new int[]{2, 3, 5}, 8));

//        System.out.println(combinationSum2(new int[]{10,1,2,7,6,1,5}, 8));

//        System.out.println(jump(new int[]{2, 3, 1, 1, 4}));

        System.out.println(permute(new int[]{1, 2, 3}));

        System.out.println(permuteUnique(new int[]{1, 1, 2}));

        System.out.println(maxSubArray(new int[]{-2, 1, -3, 4, -1, 2, 1, -5, 4}));
    }


}
