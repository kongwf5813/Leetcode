package com.owen.algorithm.v2;

import com.owen.algorithm.LinkList.ListNode;
import com.owen.algorithm.Others.Node;
import com.owen.algorithm.Tree.TreeNode;

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

    //[47].全排列II
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

    public static void setZeroes(int[][] matrix) {
        boolean needSetFirstCol = false;

        int m = matrix.length, n = matrix[0].length;
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0) {
                needSetFirstCol = true;
            }

            //将边上置为0
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }

        //处理中间
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }


        //处理第一行
        if (matrix[0][0] == 0) {
            for (int i = 0; i < n; i++) {
                matrix[0][i] = 0;
            }
        }

        //处理第一列
        if (needSetFirstCol) {
            for (int i = 0; i < m; i++) {
                matrix[i][0] = 0;
            }
        }
    }

    //[58].最后一个单词的长度
    public static int lengthOfLastWord(String s) {
        int end = s.length() - 1;
        while (end >= 0 && s.charAt(end) == ' ') {
            end--;
        }
        int start = end;

        while (start >= 0 && s.charAt(start) != ' ') {
            start--;
        }
        return end - start;
    }

    //[59].螺旋矩阵II
    public static int[][] generateMatrix(int n) {
        if (n < 0) return null;

        int[][] res = new int[n][n];
        int top = 0, bottom = n - 1, left = 0, right = n - 1, num = 1;
        while (top <= bottom && left <= right) {
            for (int col = left; col <= right; col++) {
                res[top][col] = num++;
            }
            for (int row = top + 1; row <= bottom; row++) {
                res[row][right] = num++;
            }

            if (right > left && bottom > top) {
                for (int col = right - 1; col >= left; col--) {
                    res[bottom][col] = num++;
                }

                for (int row = bottom - 1; row > top; row--) {
                    res[row][left] = num++;
                }
            }
            top++;
            right--;
            bottom--;
            left++;

        }
        return res;
    }

    //[61].旋转链表
    public static ListNode rotateRight(ListNode head, int k) {
        int size = 0;
        ListNode cur = head;
        while (cur != null) {
            size++;
            cur = cur.next;
        }
        int realK = k % size;
        if (size == 1 || size == 0 || realK == 0) {
            return head;
        }

        ListNode slow = head, fast = head;
        while (realK-- > 0) {
            fast = fast.next;
        }
        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }
        ListNode dummyHead = new ListNode(-1);
        dummyHead.next = slow.next;
        fast.next = head;
        slow.next = null;
        return dummyHead.next;
    }

    //[62].不同的路径
    public static int uniquePaths(int m, int n) {
        //dp[i][j] 有多少种路径
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 0; i < n; i++) {
            dp[0][i] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    //[63].不同的路径II
    public static int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int row = obstacleGrid.length, col = obstacleGrid[0].length;
        int[][] dp = new int[row][col];

        for (int i = 0; i < col; i++) {
            if (obstacleGrid[0][i] == 1) {
                break;
            } else {
                dp[0][i] = 1;
            }
        }
        for (int i = 0; i < row; i++) {
            if (obstacleGrid[i][0] == 1) {
                break;
            } else {
                dp[i][0] = 1;
            }
        }

        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                if (obstacleGrid[i][j] == 0) {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[row - 1][col - 1];
    }

    //[64].最小路径和
    public static int minPathSum(int[][] grid) {
        int m = grid.length;
        if (m == 0) return -1;

        int n = grid[0].length;
        //最小路径和
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < m; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int i = 1; i < n; i++) {
            dp[0][i] = dp[0][i - 1] + grid[0][i];
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }

    //[66].加一
    public static int[] plusOne(int[] digits) {
        for (int i = digits.length - 1; i >= 0; i--) {
            digits[i]++;
            digits[i] %= 10;
            //没有发生进位，直接返回
            if (digits[i] != 0) {
                return digits;
            }
        }
        //一直有进位
        digits = new int[digits.length + 1];
        digits[0] = 1;
        return digits;
    }

    //[67].二进制求和
    public static String addBinary(String a, String b) {
        int n = Math.max(a.length(), b.length());
        StringBuilder sb = new StringBuilder();
        int carry = 0;
        //最大长度，倒序遍历
        for (int i = 0; i < n; i++) {
            carry += i < a.length() ? (a.charAt(a.length() - i - 1) - '0') : 0;
            carry += i < b.length() ? (b.charAt(b.length() - i - 1) - '0') : 0;
            sb.insert(0, carry % 2);
            carry /= 2;
        }

        if (carry > 0) {
            sb.insert(0, 1);
        }

        return sb.toString();
    }

    //[69]x的平方根
    public int mySqrt(int x) {
        int left = 1, right = x;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (mid * mid == x) {
                return mid;
            } else if (mid * mid > x) {
                right = mid - 1;
            } else {
                left = mid;
            }
        }
        return left;
    }

    //[70].爬楼梯
    public static int climbStairs(int n) {
        if (n == 1) return 1;

        //dp到达i阶楼梯表示有多少种
        int[] dp = new int[n];
        dp[0] = 1;
        dp[1] = 2;
        for (int i = 2; i < n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n - 1];
    }

    //[82].删除排序链表中的重复元素II
    public static ListNode deleteDuplicates(ListNode head) {
        ListNode dummyHead = new ListNode(-1);
        dummyHead.next = head;
        ListNode cur = dummyHead;

        while (cur.next != null && cur.next.next != null) {
            if (cur.next.val == cur.next.next.val) {
                int val = cur.next.val;

                while (cur.next != null && val == cur.next.val) {
                    cur.next = cur.next.next;
                }
            } else {
                cur = cur.next;
            }
        }
        return dummyHead.next;
    }

    //[83].删除排序链表种的重复元素
    public static ListNode deleteDuplicates2(ListNode head) {
        if (head == null) return head;
        //当慢针和快针相等的时候，跳过快针当前值
        //1,1,2,3,3
        ListNode cur = head;

        while (cur.next != null) {
            if (cur.val == cur.next.val) {
                cur.next = cur.next.next;
            } else {
                cur = cur.next;
            }
        }
        return head;
    }

    //[86].分隔链表
    public static ListNode partition(ListNode head, int x) {
        ListNode largeHead = new ListNode(-1);
        ListNode large = largeHead;
        ListNode smallHead = new ListNode(-1);
        ListNode small = smallHead;
        while (head != null) {
            if (head.val < x) {
                small.next = head;
                small = small.next;
            } else {
                large.next = head;
                large = large.next;
            }
            head = head.next;
        }

        //断开指针
        large.next = null;
        small.next = largeHead.next;
        return smallHead.next;
    }

    //[88].合并两个有序数组
    public static void merge(int[] nums1, int m, int[] nums2, int n) {
        int p = m - 1, q = n - 1;
        int tail = m + n - 1;
        while (p >= 0 || q >= 0) {
            if (p == -1) {
                nums1[tail--] = nums2[q--];
            } else if (q == -1) {
                nums1[tail--] = nums1[p--];
            } else if (nums1[p] < nums2[q]) {
                nums1[tail--] = nums2[q--];
            } else {
                nums1[tail--] = nums1[p--];
            }
        }
    }

    //[89].格雷编码
    public static List<Integer> grayCode(int n) {
        //n = 3, 互为镜像，倒叙 追加1 即可
        //000
        //001
        //011
        //010
        //110
        //111
        //101
        //100
        List<Integer> res = new ArrayList<>();
        res.add(0);
        int head = 1;
        //0 -> 2
        //1 -> 4
        //2 -> 8
        for (int i = 0; i < n; i++) {
            for (int j = res.size() - 1; j >= 0; j--) {
                res.add(head + res.get(j));
            }
            head <<= 1;
        }
        return res;
    }

    //[90].子集II
    public static List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        dfsForSubsetsWithDup(nums, 0, new LinkedList<>(), res);
        return res;
    }

    private static void dfsForSubsetsWithDup(int[] nums, int start, LinkedList<Integer> select, List<List<Integer>> res) {
        res.add(new ArrayList<>(select));
        for (int i = start; i < nums.length; i++) {
            //本层决策树如果发现之前有相同的跳过
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            select.addLast(nums[i]);
            dfsForSubsetsWithDup(nums, i + 1, select, res);
            select.removeLast();
        }
    }

    //[93].复原IP地址
    public static List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        dfsForRestoreIpAddresses(s, 0, new LinkedList(), res);
        return res;
    }

    private static void dfsForRestoreIpAddresses(String s, int start, LinkedList<String> select, List<String> res) {
        if (select.size() > 4) {
            return;
        }
        if (select.size() >= 4 && start != s.length()) {
            return;
        }
        if (select.size() == 4) {
            res.add(String.join(".", select));
            return;
        }
        //010010
        for (int i = start; i < s.length(); i++) {
            String choice = s.substring(start, i + 1);
            int len = choice.length();
            //前导0直接跳过
            if (len > 1 && choice.charAt(0) == '0') {
                return;
            } else if (len > 3) {
                return;
            } else if (Integer.parseInt(choice) < 0 || Integer.parseInt(choice) > 255) {
                return;
            }
            select.addLast(choice);
            dfsForRestoreIpAddresses(s, i + 1, select, res);
            select.removeLast();
        }

    }

    //[94].二叉树的中序遍历（递归法）
    public static List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        dfsForInorderTraversal(root, res);
        return res;
    }

    private static void dfsForInorderTraversal(TreeNode root, List<Integer> res) {
        if (root == null) return;

        dfsForInorderTraversal(root.left, res);
        res.add(root.val);
        dfsForInorderTraversal(root.right, res);
    }

    //[94].二叉树的中序遍历（迭代法）
    public static List<Integer> inorderTraversal2(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        while (cur != null || !stack.isEmpty()) {
            if (cur != null) {
                stack.push(cur);
                cur = cur.left;
            } else {
                cur = stack.pop();
                res.add(cur.val);
                cur = cur.right;
            }
        }
        return res;
    }

    //[95].不同的二叉搜索树II
    public static List<TreeNode> generateTrees(int n) {
        return dfsForGenerateTrees(1, n);
    }

    private static List<TreeNode> dfsForGenerateTrees(int start, int end) {
        List<TreeNode> res = new ArrayList<>();
        if (start > end) {
            res.add(null);
            return res;
        }

        for (int i = start; i <= end; i++) {
            List<TreeNode> left = dfsForGenerateTrees(start, i - 1);
            List<TreeNode> right = dfsForGenerateTrees(i + 1, end);
            for (TreeNode l : left) {
                for (TreeNode r : right) {
                    TreeNode root = new TreeNode(i);
                    root.left = l;
                    root.right = r;
                    res.add(root);
                }
            }
        }
        return res;
    }

    //[96].不同的二叉搜索树
    public static int numTrees(int n) {
        //1 2 3 4
        //1  1
        //2  1 + 1
        //3  2 + 1 + 2
        //4  5 + 2*2 + 2*2 + 5
        //dp[4] = dp[0]* dp[3] + dp[1] * dp[2] + dp[2] * dp[1] + dp[3] * dp[0]
        //i个数，组成树的个数
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                dp[i] += dp[j] * dp[i - j - 1];
            }
        }
        return dp[n];
    }

    //[97].交错字符串
    public static boolean isInterleave(String s1, String s2, String s3) {
        //aa b c c     d bbc a   aadbbcbcac
        int m = s1.length();
        int n = s2.length();
        if (m + n != s3.length()) {
            return false;
        }
        //dp[i][j], s1为前i个， s2为前j个的时候，s3为i + j -1位置是否能够匹配上。
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i > 0) {
                    //i的意义是前i个，索引是i-1
                    dp[i][j] = dp[i][j] || dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1);
                }
                if (j > 0) {
                    //选择s2(j)，取决于选择s1(i)，s2(j-1)可以错配成功，dp[i][j-1]代表了之前两种选择:
                    //如果是最后选s1(i), s2(j)单独构成错配字符串。
                    //如果是最后选s2(j-1), s2(j-1)与s2(j)构成连续字符串，属于错配字符串。
                    dp[i][j] = dp[i][j] || dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1);
                }
            }
        }
        return dp[m][n];
    }

    //[98].验证二叉搜索树
    public static boolean isValidBST(TreeNode root) {
        return dfsForIsValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    private static boolean dfsForIsValidBST(TreeNode root, long min, long max) {
        if (root == null) {
            return true;
        }
        if (min < root.val && root.val < max) {
            return dfsForIsValidBST(root.left, min, root.val) && dfsForIsValidBST(root.right, root.val, max);
        } else {
            return false;
        }
    }

    //[100].相同的树
    public static boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p != null && q == null) return false;
        if (p == null && q != null) return false;
        if (p.val != q.val) return false;
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    //[101].对称二叉树
    public static boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return dfsIsSymmetric(root.left, root.right);
    }

    private static boolean dfsIsSymmetric(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
        if (left != null && right == null) return false;
        if (left == null && right != null) return false;
        if (left.val != right.val) return false;
        return dfsIsSymmetric(left.left, right.right) && dfsIsSymmetric(left.right, right.left);
    }

    //[102].二叉树的层序遍历
    public static List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> level = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();
                level.add(cur.val);
                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }
            }
            res.add(level);
        }
        return res;
    }

    //[103].二叉树的锯齿形层序遍历
    public static List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        List<List<Integer>> res = new ArrayList<>();

        boolean leftOrder = true;
        while (!queue.isEmpty()) {
            int size = queue.size();
            LinkedList<Integer> layer = new LinkedList<>();
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();
                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }

                if (leftOrder) layer.addLast(cur.val);
                else layer.addFirst(cur.val);
            }
            leftOrder = !leftOrder;
            res.add(layer);
        }
        return res;
    }

    //[104].二叉树的最大深度
    public static int maxDepth(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) return 1;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    //[105].从前序与中序遍历序列构造二叉树
    public static TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder.length != inorder.length) return null;

        Map<Integer, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            indexMap.put(inorder[i], i);
        }
        return dfsForBuildTree(preorder, inorder, 0, preorder.length - 1, 0, inorder.length - 1, indexMap);
    }

    private static TreeNode dfsForBuildTree(int[] preorder, int[] inorder, int ps, int pe, int is, int ie, Map<Integer, Integer> indexMap) {
        if (ps > pe || is > ie) {
            return null;
        }
        //[3,9,20,15,7] 前序
        //[9,3,15,20,7] 中序
        int val = preorder[ps];
        TreeNode root = new TreeNode(val);
        int index = indexMap.get(val);
        TreeNode left = dfsForBuildTree(preorder, inorder, ps + 1, index - is + ps, is, index - 1, indexMap);
        TreeNode right = dfsForBuildTree(preorder, inorder, index - is + ps + 1, pe, index + 1, ie, indexMap);
        root.left = left;
        root.right = right;
        return root;
    }

    //[106].从中序与后序遍历序列构造二叉树
    public static TreeNode buildTree2(int[] inorder, int[] postorder) {
        Map<Integer, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            indexMap.put(inorder[i], i);
        }
        return dfsForBuildTree2(inorder, 0, inorder.length - 1, postorder, 0, postorder.length - 1, indexMap);
    }

    private static TreeNode dfsForBuildTree2(int[] inorder, int is, int ie, int[] postorder, int ps, int pe, Map<Integer, Integer> indexMap) {
        //左根右
        //左右根
        if (is > ie || ps > pe) return null;
        int cur = postorder[pe];
        int index = indexMap.get(cur);
        TreeNode root = new TreeNode(cur);
        TreeNode left = dfsForBuildTree2(inorder, is, index - 1, postorder, ps, ps + index - is - 1, indexMap);
        TreeNode right = dfsForBuildTree2(inorder, index + 1, ie, postorder, ps + index - is, pe - 1, indexMap);
        root.left = left;
        root.right = right;
        return root;
    }

    //[107].二叉树的层序遍历II
    public static List<List<Integer>> levelOrderBottom(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        List<List<Integer>> res = new ArrayList<>();
        while (!queue.isEmpty()) {
            List<Integer> layer = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();

                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }

                layer.add(cur.val);
            }
            res.add(0, layer);
        }
        return res;
    }

    //[108].将有序数组转换为二叉搜索树
    public static TreeNode sortedArrayToBST(int[] nums) {
        return dfsForSortedArrayToBST(nums, 0, nums.length - 1);
    }

    private static TreeNode dfsForSortedArrayToBST(int[] nums, int s, int e) {
        if (s > e) return null;
        int mid = s + (e - s) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = dfsForSortedArrayToBST(nums, s, mid - 1);
        root.right = dfsForSortedArrayToBST(nums, mid + 1, e);
        return root;
    }

    //[109].有序链表转化二叉搜索树
    public static TreeNode sortedListToBST(ListNode head) {
        //左闭右开
        return dfsForBuildSortedListToBST(head, null);
    }

    private static TreeNode dfsForBuildSortedListToBST(ListNode left, ListNode right) {
        if (left == right) return null;
        ListNode mid = getMedian(left, right);
        TreeNode root = new TreeNode(mid.val);
        root.left = dfsForBuildSortedListToBST(left, mid);
        root.right = dfsForBuildSortedListToBST(mid.next, right);
        return root;
    }

    private static ListNode getMedian(ListNode left, ListNode right) {
        ListNode slow = left;
        ListNode fast = left;
        while (fast != right && fast.next != right) {
            fast = fast.next;
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    //[110].平衡二叉树
    public static boolean isBalanced(TreeNode root) {
        return getDepth(root) >= 0;
    }

    private static int getDepth(TreeNode root) {
        if (root == null) return 0;

        int leftHeight = getDepth(root.left);
        int rightHeight = getDepth(root.right);

        //高度相差超过1，说明肯定不平衡，直接返回-1，代表不平衡。
        if (leftHeight == -1 || rightHeight == -1 || Math.abs(leftHeight - rightHeight) > 1) {
            return -1;
        }
        return Math.max(getDepth(root.left), getDepth(root.right)) + 1;
    }

    //[111].二叉树的最小深度
    public static int minDepth(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) return 1;
        int leftDepth = minDepth(root.left);
        int rightDepth = minDepth(root.right);
        //如果左子树或右子树为空，代表树的深度为另一边的深度
        if (root.left == null || root.right == null) {
            return leftDepth + rightDepth + 1;
        }
        return Math.min(leftDepth, rightDepth) + 1;
    }

    //[112].路径总和
    public static boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) return false;

        if (root.left == null && root.right == null) {
            return targetSum == root.val;
        }

        return hasPathSum(root.left, targetSum - root.val)
                || hasPathSum(root.right, targetSum - root.val);
    }

    //[113].路径总和II
    public static List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> res = new ArrayList<>();
        dfsForPathSum(root, targetSum, new LinkedList<>(), res);
        return res;
    }

    private static void dfsForPathSum(TreeNode root, int sum, LinkedList<Integer> select, List<List<Integer>> res) {
        if (root == null) return;

        select.addLast(root.val);

        if (root.left == null && root.right == null) {
            if (sum == root.val) {
                res.add(new ArrayList<>(select));
            }
        }

        dfsForPathSum(root.left, sum - root.val, select, res);
        dfsForPathSum(root.right, sum - root.val, select, res);
        select.removeLast();
    }

    //[114].二叉树展开为链表
    public static void flatten(TreeNode root) {
        if (root == null) return;

        flatten(root.left);
        flatten(root.right);

        TreeNode left = root.left;
        TreeNode right = root.right;
        root.left = null;
        root.right = left;

        TreeNode cur = root;
        while (cur.right != null) {
            cur = cur.right;
        }
        cur.right = right;
    }

    //[116].填充每个节点的下一个右侧节点指针
    public static Node connect(Node root) {
        dfsForConnect(root.left, root.right);
        return root;
    }

    private static void dfsForConnect(Node left, Node right) {
        if (left == null || right == null) return;
        left.next = right;

        dfsForConnect(left.left, left.right);
        dfsForConnect(left.right, right.left);
        dfsForConnect(right.left, right.right);
    }

    //[117].填充每个节点的下一个右侧节点指针II
    public static Node connect2(Node root) {
        dfsForConnect2(root, null);
        return root;
    }

    private static void dfsForConnect2(Node left, Node right) {
        //边界情况，二叉树中间为空的需要从隔壁找到最左边的节点
        if (left == null) return;
        left.next = right;
        Node next = right;
        Node nextOne = null;
        while (next != null) {
            if (next.left != null) {
                nextOne = next.left;
                break;
            }
            if (right.right != null) {
                nextOne = next.right;
                break;
            }
            next = next.next;
        }
        if (left.right != null) {
            dfsForConnect2(left.left, left.right);
            dfsForConnect2(left.right, nextOne);
        } else {
            dfsForConnect2(left.left, nextOne);
        }
    }

    //[118].杨辉三角
    public static List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            List<Integer> layer = new ArrayList<>();
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) {
                    layer.add(1);
                } else {
                    layer.add(res.get(i - 1).get(j - 1) + res.get(i - 1).get(j));
                }
            }
            res.add(layer);
        }
        return res;
    }

    //[119].杨辉三角II
    public static List<Integer> getRow(int rowIndex) {
        List<Integer> pre = new ArrayList<>();
        for (int i = 0; i <= rowIndex; i++) {
            List<Integer> cur = new ArrayList<>();
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) {
                    cur.add(1);
                } else {
                    cur.add(pre.get(j - 1) + pre.get(j));
                }
            }
            pre = cur;
        }
        return pre;
    }

    //[120].三角形的最小路径和
    public static int minimumTotal(List<List<Integer>> triangle) {
        int size = triangle.size();
//        int[][] dp = new int[size][size];
//        dp[0][0] = triangle.get(0).get(0);
//        for (int i = 1; i < size; i++) {
//            for (int j = 0; j <= i; j++) {
//                if (j == 0) {
//                    dp[i][j] = dp[i - 1][j] + triangle.get(i).get(j);
//                } else if (j == i) {
//                    dp[i][j] = dp[i-1][j-1] + triangle.get(i).get(j);
//                } else {
//                    dp[i][j] = Math.min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle.get(i).get(j);
//                }
//            }
//        }
        int[] dp = new int[size];
        dp[0] = triangle.get(0).get(0);
        for (int i = 1; i < size; i++) {
            int pre = 0, cur;
            for (int j = 0; j <= i; j++) {
                cur = dp[j];
                if (j == 0) {
                    dp[j] = cur + triangle.get(i).get(j);
                } else if (j == i) {
                    dp[j] = pre + triangle.get(i).get(j);
                } else {
                    dp[j] = Math.min(pre, cur) + triangle.get(i).get(j);
                }
                pre = cur;
            }
        }
        int res = Integer.MAX_VALUE;
        for (int i = 0; i < size; i++) {
            res = Math.min(res, dp[i]);
        }
        return res;
    }

    //[129].求根节点到叶节点数字之和
    public static int sumNumbers(TreeNode root) {
        return dfsForSumNumbers(root, 0);
    }

    private static int dfsForSumNumbers(TreeNode root, int pre) {
        if (root == null) return 0;
        int value = 10 * pre + root.val;
        if (root.left == null && root.right == null) return value;
        return dfsForSumNumbers(root.left, value) + dfsForSumNumbers(root.right, value);
    }

    //[130].被围绕的区间
    public static void solve(char[][] board) {
        int row = board.length, col = board[0].length;
        for (int i = 0; i < row; i++) {
            dfsForSolve(board, i, 0);
            dfsForSolve(board, i, col - 1);
        }
        for (int j = 0; j < col; j++) {
            dfsForSolve(board, 0, j);
            dfsForSolve(board, row - 1, j);
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                char ch = board[i][j];
                if (ch == 'O') {
                    board[i][j] = 'X';
                } else if (ch == 'Y') {
                    board[i][j] = 'O';
                }
            }
        }
    }

    private static void dfsForSolve(char[][] board, int x, int y) {
        int row = board.length, col = board[0].length;
        if (x < 0 || y < 0
                || x >= row || y >= col
                || board[x][y] != 'O') {
            return;
        }
        board[x][y] = 'Y';
        int[][] direction = new int[][]{{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        for (int[] direct : direction) {
            int newX = x + direct[0];
            int newY = y + direct[1];
            dfsForSolve(board, newX, newY);
        }
    }

    //[131].分割回文串
    public static List<List<String>> partition(String s) {
        int len = s.length();
        boolean[][] dp = new boolean[len][len];
        for (int i = 0; i < len; i++) {
            dp[i][i] = true;
        }
        for (int i = len - 2; i >= 0; i--) {
            for (int j = i + 1; j <= len - 1; j++) {
                dp[i][j] = (dp[i + 1][j - 1] || j - i < 3) && s.charAt(i) == s.charAt(j);
            }
        }

        List<List<String>> res = new ArrayList<>();
        dfsForPartition(s, 0, new LinkedList<>(), res, dp);
        return res;
    }

    private static void dfsForPartition(String s, int start, LinkedList<String> select, List<List<String>> res, boolean[][] dp) {
        if (start == s.length()) {
            res.add(new ArrayList<>(select));
            return;
        }

        for (int i = start; i < s.length(); i++) {
            if (!dp[start][i]) {
                continue;
            }
            select.addLast(s.substring(start, i + 1));
            dfsForPartition(s, i + 1, select, res, dp);
            select.removeLast();
        }
    }

    //[133].克隆图
    public static Node cloneGraph(Node node) {
        Map<Node, Node> cloneMap = new HashMap<>();
        Map<Node, Boolean> visited = new HashMap<>();
        Node cloneRoot = new Node(node.val);
        cloneMap.put(cloneRoot, node);

        Queue<Node> queue = new LinkedList();
        queue.offer(node);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                Node cur = queue.poll();
                if (Boolean.TRUE.equals(visited.get(cur))) {
                    continue;
                }

                visited.putIfAbsent(cur, Boolean.TRUE);
                cloneMap.putIfAbsent(cur, new Node(cur.val));

                Node copy = cloneMap.get(cur);
                if (cur.neighbors != null) {
                    List<Node> cloneNeighbors = new ArrayList<>();
                    for (Node neighbor : cur.neighbors) {
                        cloneMap.putIfAbsent(neighbor, new Node(neighbor.val));
                        cloneNeighbors.add(cloneMap.get(neighbor));
                        queue.offer(neighbor);
                    }
                    copy.neighbors = cloneNeighbors;
                }
            }
        }
        return cloneRoot;
    }

    //[134].加油站
    public static int canCompleteCircuit(int[] gas, int[] cost) {
        int num = gas.length;
        int left = 0, minLeft = Integer.MAX_VALUE, start = -1;
        for (int i = 0; i < num; i++) {
            left += gas[i] - cost[i];
            if (minLeft > left) {
                minLeft = left;
                start = i;
            }
        }
        return left >= 0 ? start + 1 : -1;
    }

    //[136].只出现一次的数字
    public static int singleNumber(int[] nums) {
        int res = nums[0];
        for (int i = 1; i < nums.length; i++) {
            res ^= nums[i];
        }
        return res;
    }

    //[137].只出现一次的数字II
    public static int singleNumber2(int[] nums) {
        int res = 0;
        //32位整数
        for (int i = 0; i < 32; i++) {
            //统计每一位1的个数
            int count = 0;
            for (int j = 0; j < nums.length; j++) {
                int num = nums[j];
                count += (num >>> i) & 1;
            }
            //与3求模，要么是1 要么是0，然后每一位拼接
            res = res | (count % 3) << i;
        }
        return res;
    }

    //[138].复制带随机指针的链表
    public static Node copyRandomList(Node head) {
        if (head == null) return null;
        Map<Node, Node> copyMap = new HashMap<>();
        Node newHead = new Node(head.val);
        copyMap.put(head, newHead);
        while (head != null) {
            Node copy = copyMap.get(head);

            if (head.next != null) {
                copyMap.putIfAbsent(head.next, new Node(head.next.val));
                copy.next = copyMap.get(head.next);
            }
            if (head.random != null) {
                copyMap.putIfAbsent(head.random, new Node(head.random.val));
                copy.random = copyMap.get(head.random);
            }
            head = head.next;
        }
        return newHead;
    }

    //[139].单词拆分
    public static boolean wordBreak(String s, List<String> wordDict) {
        int len = s.length();
        //前i个字符串是否可以拆分
        boolean[] dp = new boolean[len + 1];
        dp[0] = true;
        for (int i = 1; i <= len; i++) {
            for (int j = 0; j < i; j++) {
                dp[i] = dp[j] && wordDict.contains(s.substring(j, i));
                //找到能拆分的就不需要继续了
                if (dp[i]) {
                    break;
                }
            }
        }
        return dp[len];
    }

    //[141].环形链表
    public static boolean hasCycle(ListNode head) {
        if (head == null) return false;
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;

            if (slow == fast) {
                return true;
            }
        }
        return false;
    }

    //[142].环形链表II
    public static ListNode detectCycle(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) {
                break;
            }
        }

        if (fast == null || fast.next == null) {
            return null;
        }

        slow = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }

    //[143].重排链表
    public static void reorderList(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        fast = slow.next;
        slow.next = null;
        ListNode dummyHead = new ListNode(-1), cur = fast;
        while (cur != null) {
            ListNode dummyHeadNext = dummyHead.next;
            ListNode next = cur.next;
            cur.next = dummyHeadNext;
            dummyHead.next = cur;
            cur = next;
        }

        fast = dummyHead.next;
        slow = head;
        while (fast != null) {
            ListNode slowNext = slow.next;
            ListNode fastNext = fast.next;

            fast.next = slowNext;
            slow.next = fast;

            slow = slowNext;
            fast = fastNext;
        }
    }

    //[144].二叉树的前序遍历
    public static List<Integer> preorderTraversal(TreeNode root) {
        //根左右
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode cur = stack.pop();
            res.add(cur.val);
            if (cur.right != null) {
                stack.push(cur.right);
            }
            if (cur.left != null) {
                stack.push(cur.left);
            }
        }
        return res;
    }

    //[145].二叉树的后序遍历
    public static List<Integer> postorderTraversal(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        List<Integer> res = new ArrayList<>();
        //关键在于右子树的后序遍历规律
        //先一直遍历右子树，并丢入栈中，直到为空，然后弹出一个节点，再继续遍历该节点的右子树。
        TreeNode cur = root;
        while (cur != null || !stack.isEmpty()) {
            if (cur != null) {
                //倒叙插入节点
                res.add(0, cur.val);

                stack.push(cur);
                cur = cur.right;
            } else {
                cur = stack.pop();
                cur = cur.left;
            }
        }
        return res;
    }

    //[146].LRU 缓存机制
    class LRUCache {

        LinkedHashMap<Integer, Integer> map;
        int capacity;

        public LRUCache(int capacity) {
            this.map = new LinkedHashMap<>();
            this.capacity = capacity;
        }

        public int get(int key) {
            if (!map.containsKey(key)) return -1;
            int value = map.get(key);
            map.remove(key);
            map.put(key, value);
            return value;
        }

        public void put(int key, int value) {
            if (map.containsKey(key)) {
                map.remove(key);
                map.put(key, value);
                return;
            }

            if (map.size() >= capacity) {
                map.remove(map.keySet().iterator().next());
            }
            map.put(key, value);
        }
    }

    //[147].对链表进行插入排序
    public static ListNode insertionSortList(ListNode head) {
        ListNode dummyHead = new ListNode(Integer.MIN_VALUE);
        dummyHead.next = head;
        //相等没必要比较
        ListNode cur = head.next, lastSorted = head;
        while (cur != null) {
            if (lastSorted.val <= cur.val) {
                lastSorted = lastSorted.next;
            } else {
                ListNode pre = dummyHead;
                //找到带插入节点选择前面一个节点
                while (pre.next.val <= cur.val) {
                    pre = pre.next;
                }
                lastSorted.next = cur.next;
                cur.next = pre.next;
                pre.next = cur;
            }
            cur = lastSorted.next;
        }
        return dummyHead.next;
    }

    //[148].排序链表
    public static ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode slow = head, fast = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }

        ListNode mid = slow.next;
        slow.next = null;
        ListNode first = sortList(head);
        ListNode second = sortList(mid);

        //合并两个有序链表
        ListNode dummyHead = new ListNode(0), cur = dummyHead;
        while (first != null && second != null) {
            if (first.val < second.val) {
                cur.next = first;
                first = first.next;
                cur = cur.next;
            } else {
                cur.next = second;
                second = second.next;
                cur = cur.next;
            }
        }
        cur.next = (first == null) ? second : first;
        return dummyHead.next;
    }

    //[150].逆波兰表达式求值
    public static int evalRPN(String[] tokens) {
        Stack<String> stack = new Stack<>();
        for (int i = 0; i < tokens.length; i++) {
            String token = tokens[i];
            if (token.equals("+")
                    || token.equals("-")
                    || token.equals("*")
                    || token.equals("/")) {
                int num2 = Integer.parseInt(stack.pop());
                int num1 = Integer.parseInt(stack.pop());
                int res = 0;
                if (token.equals("+")) {
                    res = num1 + num2;
                } else if (token.equals("-")) {
                    res = num1 - num2;
                } else if (token.equals("*")) {
                    res = num1 * num2;
                } else {
                    res = num1 / num2;
                }
                stack.push(String.valueOf(res));
            } else {
                stack.push(token);
            }
        }
        return stack.isEmpty() ? 0 : Integer.parseInt(stack.pop());
    }

    //[151].翻转字符串里的单词
    public static String reverseWords(String s) {
        String[] arr = s.split("[\\s\\u00A0]+");
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < arr.length; i++) {
            sb.insert(0, arr[i] + ' ');
        }
        if (sb.length() > 0) sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
    }

    //[152].乘积最大子数组
    public static int maxProduct(int[] nums) {
        int n = nums.length;
        //dp[i][0], 0到i子数组的最大值
        //dp[i][1], 0到i子数组的最小值
        int[][] dp = new int[n][2];
        dp[0][0] = nums[0];
        dp[0][1] = nums[0];
        int max = Integer.MIN_VALUE;
        for (int i = 1; i < n; i++) {
            if (nums[i] > 0) {
                dp[i][0] = Math.max(dp[i - 1][0] * nums[i], nums[i]);
                dp[i][1] = Math.min(dp[i - 1][1] * nums[i], nums[i]);
            } else {
                dp[i][0] = Math.max(dp[i - 1][1] * nums[i], nums[i]);
                dp[i][1] = Math.min(dp[i - 1][0] * nums[i], nums[i]);
            }
            max = Math.max(max, dp[i][0]);
        }
        return max;
    }

    //[153].寻找旋转排序数组中的最小值
    public static int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return nums[left];
    }

    //[154].寻找旋转排序数组中的最小值II
    public static int findMin2(int[] nums) {
        int left = 0, right = nums.length - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;
            //01111只能代表mid到right的区间一定是都是相等的，可以砍掉一个右边界值
            if (nums[mid] == nums[right]) {
                right--;
            } else if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                //mid小于右边值，说明可能mid可能是个最小值，不能mid-1
                right = mid;
            }
        }
        return nums[left];
    }

    //[160].相交链表
    public static ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        ListNode curA = headA, curB = headB;
        while (curA != curB) {
            if (curA == null) {
                curA = headB;
            } else {
                curA = curA.next;
            }

            if (curB == null) {
                curB = headA;
            } else {
                curB = curB.next;
            }
        }
        return curA;
    }

    //[162].寻找峰值
    public static int findPeakElement(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            //下降区间，往左边找峰值
            if (nums[mid] > nums[mid + 1]) {
                //mid可能就是个峰值
                right = mid;
            } else {
                //上升区间，nums[mid] != nums[mid+1],表示mid +1才可能能是峰值，往右边找峰值
                left = mid + 1;
            }
        }
        return left;
    }

    //[165].比较版本号
    public static int compareVersion(String version1, String version2) {
        String[] a = version1.split("\\.");
        String[] b = version2.split("\\.");
        int maxLen = Math.max(a.length, b.length);
        for (int i = 0; i < maxLen; i++) {
            int x = i >= a.length ? 0 : Integer.parseInt(a[i]);
            int y = i >= b.length ? 0 : Integer.parseInt(b[i]);
            if (x > y) {
                return 1;
            } else if (x < y) {
                return -1;
            }
        }
        return 0;
    }

    //[166].分数到小数
    public static String fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) return "0";
        StringBuilder res = new StringBuilder();
        if (numerator < 0 ^ denominator < 0) {
            res.append('-');
        }

        long num = Math.abs(Long.valueOf(numerator));
        long div = Math.abs(Long.valueOf(denominator));
        res.append(num / div);
        long remain = num % div;
        if (remain == 0) return res.toString();
        res.append('.');

        Map<Long, Integer> pos = new HashMap<>();
        while (remain != 0) {
            if (pos.containsKey(remain)) {
                int p = pos.get(remain);
                res.insert(p, '(');
                res.append(')');
                break;
            } else {
                pos.put(remain, res.length());
                num = remain * 10;
                res.append(num / div);
                remain = num % div;
            }
        }
        return res.toString();

    }

    //[179].最大数
    public static String largestNumber(int[] nums) {
        int len = nums.length;
        String[] arr = new String[len];
        for (int i = 0; i < nums.length; i++) {
            arr[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(arr, (a, b) -> (b + a).compareTo(a + b));
        //[0,0]
        if (arr[0].equals("0")) return "0";
        StringBuilder sb = new StringBuilder();
        for (String str : arr) {
            sb.append(str);
        }
        return sb.toString();
    }

    //[187].重复的DNA序列
    public static List<String> findRepeatedDnaSequences(String s) {
        if (s.length() <= 10) return null;

        Set<String> res = new HashSet<>();
        Set<String> unique = new HashSet<>();
        for (int i = 9; i < s.length(); i++) {
            String sub = s.substring(i - 9, i + 1);
            if (unique.contains(sub)) {
                res.add(sub);
            }
            unique.add(sub);
        }
        return new ArrayList<>(res);
    }

    //[201].数字范围按位与
    public static int rangeBitwiseAnd(int left, int right) {
        //最长公共前缀即可
        //记录右移次数
        int count = 0;
        while (left != right) {
            left >>= 1;
            right >>= 1;
            count++;
        }
        return left << count;
    }

    //[203].移除链表元素
    public ListNode removeElements(ListNode head, int val) {
        ListNode dummyHead = new ListNode(-1);
        dummyHead.next = head;
        ListNode pre = dummyHead, cur = head;
        while (cur != null) {
            ListNode next = cur.next;
            if (cur.val == val) {
                pre.next = next;
            } else {
                pre = cur;
            }
            cur = next;
        }
        return dummyHead.next;
    }

    //[206].反转链表
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode last = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return last;
    }

    //[198].打家劫舍
    public static int rob(int[] nums) {
        //反方向定义打劫，从i到最后，打劫的最大金额
        //dp[i] = Math.max(dp[i+2]+ nums[i], dp[i+1]);
//        int n = nums.length;
//        int[] dp = new int[n + 2];
//        for (int i = n - 1; i >= 0; i--) {
//            dp[i] = Math.max(dp[i + 2] + nums[i], dp[i + 1]);
//        }
//        return dp[0];
        //状态压缩
        int n = nums.length;
        int dp_i_1 = 0, dp_i_2 = 0, dp_i = 0;
        for (int i = n - 1; i >= 0; i--) {
            dp_i = Math.max(dp_i_2 + nums[i], dp_i_1);
            //从大到小
            dp_i_2 = dp_i_1;
            dp_i_1 = dp_i;
        }
        return dp_i;
    }

    public static int rob2(int[] nums) {
        if (nums.length == 1) return nums[0];
        return Math.max(dfsForRob(nums, 0, nums.length - 2),
                dfsForRob(nums, 1, nums.length - 1));
    }

    private static int dfsForRob(int[] nums, int s, int e) {
        int dp_i = 0, dp_i_2 = 0, dp_i_1 = 0;
        for (int i = e; i >= s; i--) {
            dp_i = Math.max(dp_i_2 + nums[i], dp_i_1);
            dp_i_2 = dp_i_1;
            dp_i_1 = dp_i;
        }
        return dp_i;
    }

    //[]
    public static List<Integer> rightSideView(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        List<Integer> res = new ArrayList<>();
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                if (i == size - 1) {
                    res.add(node.val);
                }
                if (node.left != null) {
                    queue.offer(node.left);
                }

                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
        }
        return res;
    }

    //[200].岛屿的数量
    public static int numIslands(char[][] grid) {
        int row = grid.length, col = grid[0].length;
        int res = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == '1') {
                    dfsForNumIslands(grid, i, j);
                    res++;
                }
            }
        }
        return res;
    }

    private static void dfsForNumIslands(char[][] grid, int i, int j) {
        int row = grid.length, col = grid[0].length;
        if (i >= row || i < 0 || j < 0 || j >= col || grid[i][j] != '1') {
            return;
        }
        grid[i][j] = '2';
        dfsForNumIslands(grid, i + 1, j);
        dfsForNumIslands(grid, i - 1, j);
        dfsForNumIslands(grid, i, j + 1);
        dfsForNumIslands(grid, i, j - 1);
    }

    //[207]
    public static boolean canFinish(int numCourses, int[][] prerequisites) {
        // 1 <- 2,2 <- 3
        int[] inDegrees = new int[numCourses];
        for (int[] pre : prerequisites) {
            inDegrees[pre[0]]++;
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int course = 0; course < inDegrees.length; course++) {
            if (inDegrees[course] == 0) {
                queue.offer(course);
            }
        }

        int count = 0;
        while (!queue.isEmpty()) {
            int course = queue.poll();
            count++;
            for (int[] pre : prerequisites) {
                if (course != pre[1]) continue;

                inDegrees[pre[0]]--;
                if (inDegrees[pre[0]] == 0) {
                    queue.offer(pre[0]);
                }
            }
        }
        return count == numCourses;
    }

    //[208]
    class Trie {
        Trie[] next;
        boolean isEnd;

        /**
         * Initialize your data structure here.
         */
        public Trie() {
            next = new Trie[26];
            isEnd = false;
        }

        /**
         * Inserts a word into the trie.
         */
        public void insert(String word) {
            Trie root = this;
            char[] arr = word.toCharArray();
            for (char ch : arr) {
                Trie next = root.next[ch - 'a'];
                if (next == null) {
                    next = new Trie();
                    root.next[ch - 'a'] = next;
                }
                root = next;
            }
            root.isEnd = true;
        }

        /**
         * Returns if the word is in the trie.
         */
        public boolean search(String word) {
            Trie root = this;
            for (char ch : word.toCharArray()) {
                Trie next = root.next[ch - 'a'];
                if (next == null) return false;

                root = next;
            }
            return root.isEnd;
        }

        /**
         * Returns if there is any word in the trie that starts with the given prefix.
         */
        public boolean startsWith(String prefix) {
            Trie root = this;
            for (char ch : prefix.toCharArray()) {
                Trie next = root.next[ch - 'a'];
                if (next == null) return false;
                root = next;
            }
            return true;
        }
    }

    //[210]
    public static int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] res = new int[numCourses];
        int[] inDegrees = new int[numCourses];
        for (int[] pre : prerequisites) {
            inDegrees[pre[0]]++;
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int course = 0; course < inDegrees.length; course++) {
            if (inDegrees[course] == 0) {
                queue.offer(course);
            }
        }
        int index = 0;
        while (!queue.isEmpty()) {
            int course = queue.poll();
            res[index++] = course;
            for (int[] pre : prerequisites) {
                if (pre[1] != course) continue;

                inDegrees[pre[0]]--;
                if (inDegrees[pre[0]] == 0) {
                    queue.offer(pre[0]);
                }
            }
        }
        return index == numCourses ? res : new int[0];
    }

    //[209]
    public static int minSubArrayLen(int target, int[] nums) {
        //7     2,3,1,2,4,3
        int right = 0, left = 0, sum = 0, res = Integer.MAX_VALUE;
        while (right < nums.length) {
            sum += nums[right];
            right++;
            //提前多加了1
            while (sum >= target) {
                //提前多减1
                left++;
                sum -= nums[left];

                res = Math.min(res, right - left);
            }
        }
        return res == Integer.MAX_VALUE ? 0 : res;
    }

    //[217]
    public static boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (set.contains(num)) {
                return true;
            }
            set.add(num);
        }
        return false;
    }

    //[219]
    public static boolean containsNearbyDuplicate(int[] nums, int k) {
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            if (set.contains(num)) return true;
            set.add(num);
            if (set.size() > k) {
                set.remove(nums[i - k]);
            }
        }
        return false;
    }

    //[215]
    public static int findKthLargest(int[] nums, int k) {
        return 0;
    }

    //[216]
    public static List<List<Integer>> combinationSum3(int k, int n) {
        //选择1-9,边界 size = k && sum == n
        List<List<Integer>> res = new ArrayList<>();
        dfsForCombinationSum3(k, n, 1, new LinkedList<>(), res);
        return res;
    }

    private static void dfsForCombinationSum3(int k, int target, int start, LinkedList<Integer> select, List<List<Integer>> res) {
        if (select.size() == k) {
            if (target == 0) {
                res.add(new ArrayList<>(select));
            }
            return;
        }
        for (int num = start; num <= 9; num++) {
            select.addLast(num);
            dfsForCombinationSum3(k, target - num, num + 1, select, res);
            select.removeLast();
        }
    }

    //[220]
    public static boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        TreeSet<Long> set = new TreeSet<>();
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            Long ceiling = set.ceiling((long) num);
            Long floor = set.floor((long) num);
            if (ceiling != null && ceiling - (long) num <= (long) t) {
                return true;
            }
            if (floor != null && (long) num - floor <= (long) t) {
                return true;
            }
            set.add((long) num);
            if (set.size() > k) {
                set.remove(nums[i - k]);
            }
        }
        return false;
    }

    //[221]
    public static int maximalSquare(char[][] matrix) {
        int row = matrix.length, col = matrix[0].length;
        int[][] dp = new int[row][col];
        //dp[i][j] 0,0 到i,j的最大正方形的边长
        //dp[i][j] = Math.max(dp[i-1][j], dp[i][j])

        int maxSide = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(dp[i - 1][j], Math.min(dp[i - 1][j - 1], dp[i][j - 1])) + 1;
                    }
                    maxSide = Math.max(maxSide, dp[i][j]);
                }
            }
        }
        return maxSide * maxSide;
    }

    //[222]
    public static int countNodes(TreeNode root) {
        if (root == null) return 0;
        int leftLevel = countLevel(root.left);
        int rightLevel = countLevel(root.right);

        if (leftLevel == rightLevel) {
            //右子树+ (根节点 + 左子树)
            return countNodes(root.right) + (1 << leftLevel);
        } else {
            return countNodes(root.left) + (1 << rightLevel);
        }
    }

    private static int countLevel(TreeNode root) {
        if (root == null) return 0;
        int level = 0;
        while (root != null) {
            level++;
            root = root.left;
        }
        return level;
    }

    //[226]
    public static TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;

        invertTree(root.left);
        invertTree(root.right);
        return root;
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
//
//        int[] arr = new int[]{3,2,1};
//        nextPermutation(arr);
//        System.out.println(Arrays.toString(arr));
//
//        arr = new int[]{4,5,2,6,3,1};
//        nextPermutation(arr);
//        System.out.println(Arrays.toString(arr));
//
//        arr = new int[]{1,1};
//        nextPermutation(arr);
//        System.out.println(Arrays.toString(arr));
//
//        System.out.println(searchInsert(new int[]{1,3,5,6}, 5));
//        System.out.println(searchInsert(new int[]{1,3,5,6}, 2));
//        System.out.println(searchInsert(new int[]{1,3,5,6}, 7));
//        System.out.println(searchInsert(new int[]{1,3,5,6}, 0));
//
//        System.out.println(isValidSudoku(new char[][]{{'8','3','.','.','7','.','.','.','.'},
//                {'6','.','.','1','9','5','.','.','.'},
//                {'.','9','8','.','.','.','.','6','.'},
//                {'8','.','.','.','6','.','.','.','3'},
//                {'4','.','.','8','.','3','.','.','1'},
//                {'7','.','.','.','2','.','.','.','6'},
//                {'.','6','.','.','.','.','2','8','.'},
//                {'.','.','.','4','1','9','.','.','5'},
//                {'.','.','.','.','8','.','.','7','9'}}));
//
//        System.out.println(countAndSay(7));
//
//        System.out.println(combinationSum(new int[]{2, 3, 6, 7}, 7));
//        System.out.println(combinationSum(new int[]{2, 3, 5}, 8));
//
//        System.out.println(combinationSum2(new int[]{10,1,2,7,6,1,5}, 8));
//
//        System.out.println(jump(new int[]{2, 3, 1, 1, 4}));
//
//        System.out.println(permute(new int[]{1, 2, 3}));
//
//        System.out.println(permuteUnique(new int[]{1, 1, 2}));
//
//        System.out.println(maxSubArray(new int[]{-2, 1, -3, 4, -1, 2, 1, -5, 4}));
//
//        [46].全排列
//        System.out.println(permute(new int[]{1, 2, 3}));
//
//        [47].全排列II
//        System.out.println(permuteUnique(new int[]{1, 1, 2}));
//
//        [53].最大子序和
//        System.out.println(maxSubArray(new int[]{-2, 1, -3, 4, -1, 2, 1, -5, 4}));
//
//        [58].最后一个单词的长度
//        System.out.println(lengthOfLastWord("Hello word "));
//
//        [59].螺旋矩阵II
//        System.out.println(generateMatrix(3));
//
//        [61].旋转链表
//        ListNode f = new ListNode(1);
//        f.next = new ListNode(2);
//        f.next.next = new ListNode(3);
//        f.next.next.next = new ListNode(4);
//        f.next.next.next.next = new ListNode(5);
//        ListNode res = rotateRight(f, 2);

//
//        [62].不同的路径
//        System.out.println(uniquePaths(3, 7));
//        System.out.println(uniquePaths(3, 2));
//        System.out.println(uniquePaths(3, 3));
//        [63].不同的路径II
//        System.out.println(uniquePathsWithObstacles(new int[][]{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}));

//        System.out.println(minPathSum(new int[][]{{1, 3, 1}, {1, 5, 1}, {4, 2, 1}}));
//
//        System.out.println(Arrays.toString(plusOne(new int[]{0})));
//        System.out.println(restoreIpAddresses("010010"));
//
//        System.out.println(isInterleave("aabcc", "dbbca", "aadbbcbcac"));
//        System.out.println(isInterleave("aabcc", "dbbca", "aadbbbaccc"));
//        System.out.println(isInterleave("", "", "1"));
//
//        TreeNode tree103 = new TreeNode(3);
//        tree103.left = new TreeNode(9);
//        tree103.right = new TreeNode(20);
//        tree103.left.left = new TreeNode(8);
//        tree103.left.left.left = new TreeNode(18);
//        tree103.right.left = new TreeNode(15);
//        tree103.right.right = new TreeNode(7);
//        tree103.right.left.left = new TreeNode(6);
//        tree103.right.left.right = new TreeNode(22);
//        System.out.println(zigzagLevelOrder(tree103));
//
//        TreeNode tree107 = new TreeNode(3);
//        tree107.left = new TreeNode(9);
//        tree107.right = new TreeNode(20);
//        tree107.left.left = new TreeNode(15);
//        tree107.right.right =  new TreeNode(7);;
//        System.out.println(levelOrderBottom(tree107));
//
//        TreeNode tree113 = new TreeNode(5);
//        tree113.left = new TreeNode(4);
//        tree113.right = new TreeNode(8);
//        tree113.left.left = new TreeNode(11);
//        tree113.left.left.left = new TreeNode(7);
//        tree113.left.left.right = new TreeNode(2);
//        tree113.right.left = new TreeNode(13);
//        tree113.right.right = new TreeNode(4);
//        tree113.right.right.left = new TreeNode(5);
//        tree113.right.right.right = new TreeNode(1);
//        System.out.println(pathSum(tree113, 22));
//
//        TreeNode tree114 = new TreeNode(1);
//        tree114.left = new TreeNode(2);
//        tree114.right = new TreeNode(5);
//        tree114.left.left = new TreeNode(3);
//        tree114.left.right = new TreeNode(4);
//        tree114.right.right = new TreeNode(6);
//        flatten(tree114);
//        System.out.println();
//        System.out.println(generate(6));
//
//        System.out.println(getRow(6));
//
//        List<List<Integer>> triangle = new ArrayList<>();
//        triangle.add(Arrays.asList(2));
//        triangle.add(Arrays.asList(3, 4));
//        triangle.add(Arrays.asList(6, 5, 7));
//        triangle.add(Arrays.asList(4, 1, 8, 3));
//        System.out.println(minimumTotal(triangle));
//
//        char[][] board = new char[][]{{'X', 'X', 'X', 'X'}, {'X', 'O', 'O', 'X'}, {'X', 'X', 'O', 'X'}, {'X', 'O', 'X', 'X'}};
//        solve(board);
//        char[][] board2 = new char[][]{{'O'}};
//        solve(board2);
//
//        [131].分割回文串
//        System.out.println(partition("aba"));
//        System.out.println(partition("aab"));
//        System.out.println(partition("aabbaacbc"));
//
//
//        System.out.println(addBinary("11", "1"));
//        System.out.println(addBinary("1010", "1011"));
//
//        System.out.println(climbStairs(3));
//
//        int[] res = new int[]{1, 2, 3, 0, 0, 0};
//        merge(res, 3, new int[]{2, 5, 6}, 3);
//        System.out.println(Arrays.toString(res));
//
//        System.out.println(grayCode(3));
//
//        System.out.println(subsetsWithDup(new int[]{1, 2, 2}));
//
//        System.out.println(restoreIpAddresses("101023"));
//        System.out.println(restoreIpAddresses("25525511135"));
//        System.out.println(restoreIpAddresses(""));
//
//        System.out.println(singleNumber2(new int[]{2, 3, 2, 3, 2, 3, 9}));
//
//        System.out.println(wordBreak("catsandog", Arrays.asList("cats", "dog", "sand", "and", "cat")));
//        System.out.println(wordBreak("applepenapple", Arrays.asList("apple", "pen")));
//
//        ListNode f = new ListNode(1);
//        f.next = new ListNode(2);
//        f.next.next = new ListNode(3);
//        f.next.next.next = new ListNode(4);
//        f.next.next.next.next = new ListNode(5);
//        reorderList(f);
//        System.out.println();
//
//        System.out.println(evalRPN(new String[]{"2", "1", "+", "3", "*"}));
//        System.out.println(rangeBitwiseAnd(5, 7));
//        System.out.println(rob(new int[]{2, 7, 9, 3, 1}));
//        System.out.println(rob(new int[]{1, 2, 3, 1}));
//        System.out.println(rob2(new int[]{2, 7, 9, 3, 1}));
//        System.out.println(Arrays.toString(findOrder(4, new int[][]{{1, 0}, {2, 0}, {3, 1}, {3, 2}})));
//        System.out.println(Arrays.toString(findOrder(2, new int[][]{{1, 0}, {0, 1}})));
//        System.out.println(minSubArrayLen(7, new int[]{2, 3, 1, 2, 4, 3}));
//        System.out.println(minSubArrayLen(4, new int[]{1,4,4}));
//
//        System.out.println(combinationSum3(3, 9));
    }
}