package com.owen.algorithm.v3;

import com.owen.algorithm.Tree;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class AllOfThem {
    public static class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        public ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    public static class TreeNode {
        public int val;
        public TreeNode left;
        public TreeNode right;

        public TreeNode(int x) {
            val = x;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public static class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;
        public Node random;
        public List<Node> neighbors;
        public List<Node> children;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }

    private class TrieNode {
        TrieNode[] children;
        boolean isEnd;

        public TrieNode() {
            children = new TrieNode[26];
            isEnd = false;
        }
    }

    public static class Solution427 {
        public static class Node {
            public boolean val;
            public boolean isLeaf;
            public Node topLeft;
            public Node topRight;
            public Node bottomLeft;
            public Node bottomRight;

            public Node() {
                this.val = false;
                this.isLeaf = false;
                this.topLeft = null;
                this.topRight = null;
                this.bottomLeft = null;
                this.bottomRight = null;
            }

            public Node(boolean val, boolean isLeaf) {
                this.val = val;
                this.isLeaf = isLeaf;
                this.topLeft = null;
                this.topRight = null;
                this.bottomLeft = null;
                this.bottomRight = null;
            }

            public Node(boolean val, boolean isLeaf, Node topLeft, Node topRight, Node bottomLeft, Node bottomRight) {
                this.val = val;
                this.isLeaf = isLeaf;
                this.topLeft = topLeft;
                this.topRight = topRight;
                this.bottomLeft = bottomLeft;
                this.bottomRight = bottomRight;
            }
        }
    }

    class Difference {
        int[] diffSum;

        public Difference(int[] nums) {
            int n = nums.length;
            diffSum = new int[n];
            int temp = 0;
            for (int i = 0; i < n; i++) {
                diffSum[i] = nums[i] - temp;
                temp = nums[i];
            }
        }

        public void insert(int i, int j, int num) {
            diffSum[i] += num;
            //数组会超过
            if (j + 1 < diffSum.length) {
                diffSum[j + 1] -= num;
            }
        }

        public int[] result() {
            int[] nums = new int[diffSum.length];
            int sum = 0;
            for (int i = 0; i < diffSum.length; i++) {
                sum += diffSum[i];
                nums[i] = sum;
            }
            return nums;
        }
    }

    //[1].两数之和
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> mapIndex = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            if (mapIndex.containsKey(target - num)) {
                return new int[]{mapIndex.get(target - num), i};
            }
            mapIndex.put(num, i);
        }
        return new int[0];
    }

    //[2].两数相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int capacity = 0;
        ListNode dummyHead = new ListNode(-1);
        ListNode cur = dummyHead;
        while (l1 != null || l2 != null || capacity != 0) {
            int v1 = 0;
            if (l1 != null) {
                v1 = l1.val;
                l1 = l1.next;
            }
            int v2 = 0;
            if (l2 != null) {
                v2 = l2.val;
                l2 = l2.next;
            }
            int val = v1 + v2 + capacity;
            cur.next = new ListNode(val % 10);
            cur = cur.next;
            capacity = val / 10;
        }
        return dummyHead.next;
    }

    //[3].无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> window = new HashMap<>();
        int left = 0, right = 0, res = 0;
        while (right < s.length()) {
            char r = s.charAt(right);
            right++;
            window.put(r, window.getOrDefault(r, 0) + 1);
            while (window.get(r) > 1) {
                char l = s.charAt(left);
                window.put(l, window.get(l) - 1);
                left++;
            }
            //窗口扩大的时候求最长子串
            res = Math.max(res, right - left);
        }
        return res;
    }

    //[5].最长回文子串
    public String longestPalindrome(String s) {
        int n = s.length();
        if (n == 0) return "";

        //注意只要字符串非空，一定有一个单字符必定是回文，默认最小值为1
        int maxLen = 1, start = 0;
        boolean[][] dp = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
        }
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                dp[i][j] = s.charAt(i) == s.charAt(j) && (j - i < 3 || dp[i + 1][j - 1]);
                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    start = i;
                }
            }
        }
        return s.substring(start, start + maxLen);
    }

    //[6].Z 字形变换
    public String convert(String s, int numRows) {
        if (numRows == 0) return "";
        if (numRows == 1) return s;
        StringBuilder[] sbs = new StringBuilder[numRows];
        for (int i = 0; i < numRows; i++) {
            sbs[i] = new StringBuilder();
        }
        int index = 0;
        boolean flag = false;
        for (int i = 0; i < s.length(); i++) {
            if (index == numRows - 1) {
                flag = true;
            } else if (index == 0) {
                flag = false;
            }
            sbs[index].append(s.charAt(i));
            if (flag) {
                index--;
            } else {
                index++;
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < numRows; i++) {
            sb.append(sbs[i].toString());
        }
        return sb.toString();
    }

    //[7].整数反转
    public int reverse(int x) {
        int revert = 0;
        while (x != 0) {
            int a = x % 10;
            if (revert > Integer.MAX_VALUE / 10 || (revert == Integer.MAX_VALUE / 10 && a > 7))
                return 0;
            if (revert < Integer.MIN_VALUE / 10 || (revert == Integer.MIN_VALUE / 10 && a < -8))
                return 0;

            revert = revert * 10 + a;
            x /= 10;
        }
        return revert;
    }

    //[8].字符串转换整数 (atoi)
    public int myAtoi(String s) {
        boolean begin = true;
        int n = s.length();
        int sign = 1, res = 0;
        for (int i = 0; i < n; i++) {
            char ch = s.charAt(i);
            if (begin && ch == ' ') {
                continue;
            } else if (begin && ch == '-') {
                sign = -1;
                begin = false;
            } else if (begin && ch == '+') {
                sign = 1;
                begin = false;
            } else if (Character.isDigit(ch)) {
                int a = ch - '0';
                if (res < Integer.MIN_VALUE / 10 || (res == Integer.MIN_VALUE / 10 && a < Integer.MIN_VALUE % 10))
                    return Integer.MIN_VALUE;
                if (res > Integer.MAX_VALUE / 10 || (res == Integer.MAX_VALUE / 10 && a > Integer.MAX_VALUE % 10))
                    return Integer.MAX_VALUE;
                res = res * 10 + sign * a;
                begin = false;
            } else {
                break;
            }
        }
        return res;
    }

    //[9].回文数
    public boolean isPalindrome(int x) {
        if (x < 0) return false;
        int remain = x, res = 0;
        while (remain != 0) {
            int a = remain % 10;
            res = res * 10 + a;
            remain /= 10;
        }
        return x == res;
    }

    //[11].盛最多水的容器
    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1, area = 0;
        while (left < right) {
            area = Math.max(area, Math.min(height[right], height[left]) * (right - left));
            //那边最短，往里面缩
            if (height[left] > height[right]) {
                right--;
            } else {
                left++;
            }
        }
        return area;
    }

    //[12].整数转罗马数字
    public String intToRoman(int num) {
        Map<Integer, String> index = new LinkedHashMap<Integer, String>() {{
            put(1000, "M");
            put(900, "CM");
            put(500, "D");
            put(400, "CD");
            put(100, "C");
            put(90, "XC");
            put(50, "L");
            put(40, "XL");
            put(10, "X");
            put(9, "IX");
            put(5, "V");
            put(4, "IV");
            put(1, "I");
        }};
        int remain = num;
        StringBuilder sb = new StringBuilder();
        for (int number : index.keySet()) {
            //从大到小
            while (remain >= number) {
                sb.append(index.get(number));
                remain -= number;
            }
            if (remain == 0) {
                break;
            }
        }
        return sb.toString();
    }

    //[13].罗马数字转整数
    public int romanToInt(String s) {
        Map<Character, Integer> symbolValues = new HashMap<Character, Integer>() {{
            put('I', 1);
            put('V', 5);
            put('X', 10);
            put('L', 50);
            put('C', 100);
            put('D', 500);
            put('M', 1000);
        }};
        int ans = 0;
        for (int i = 0; i < s.length(); i++) {
            int cur = symbolValues.get(s.charAt(i));
            if (i + 1 < s.length() && cur < symbolValues.get(s.charAt(i + 1))) {
                ans -= cur;
            } else {
                ans += cur;
            }
        }
        return ans;
    }

    //[14].最长公共前缀
    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }
        int length = strs[0].length();
        for (int i = 0; i < length; i++) {
            char ch = strs[0].charAt(i);
            for (int j = 1; j < strs.length; j++) {
                if (i == strs[j].length() || strs[j].charAt(i) != ch) {
                    return strs[0].substring(0, i);
                }
            }
        }
        return strs[0];
    }

    //[17].电话号码的字母组合
    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        if (digits.length() == 0) return res;
        String[] numbers = new String[]{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        backtraceForLetterCombinations(digits, 0, res, new StringBuilder(), numbers);
        return res;
    }

    private void backtraceForLetterCombinations(String digits, int s, List<String> res, StringBuilder sb, String[] numbers) {
        if (s == digits.length()) {
            res.add(sb.toString());
            return;
        }
        char ch = digits.charAt(s);
        for (char choice : numbers[ch - '0'].toCharArray()) {
            sb.append(choice);
            backtraceForLetterCombinations(digits, s + 1, res, sb, numbers);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    //[19].删除链表的倒数第 N 个结点
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode slow = head, fast = head;
        int count = 0;
        while (count < n) {
            fast = fast.next;
            count++;
        }
        ListNode pre = null;
        while (fast != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next;
        }

        //第一个节点
        if (pre == null) {
            pre = slow.next;
            slow.next = null;
            return pre;
        } else {
            pre.next = slow.next;
            return head;
        }
    }

    //[20].有效的括号
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == ']') {
                if (stack.isEmpty() || stack.pop() != '[')
                    return false;
            } else if (ch == '}') {
                if (stack.isEmpty() || stack.pop() != '{')
                    return false;
            } else if (ch == ')') {
                if (stack.isEmpty() || stack.pop() != '(')
                    return false;
            } else {
                stack.push(ch);
            }
        }
        return stack.isEmpty();
    }

    //[21].合并两个有序链表
    public ListNode mergeTwoList(ListNode first, ListNode second) {
        ListNode dummyHead = new ListNode(-1), p = dummyHead;
        ListNode p1 = first, p2 = second;
        while (p1 != null && p2 != null) {
            if (p1.val < p2.val) {
                p.next = p1;
                p1 = p1.next;
            } else {
                p.next = p2;
                p2 = p2.next;
            }
            p = p.next;
        }

        if (p1 != null) {
            p.next = p1;
        }
        if (p2 != null) {
            p.next = p2;
        }
        return dummyHead.next;
    }

    //[22].括号生成
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        if (n <= 0) return res;
        backtraceForGenerateParenthesis(n, n, res, new StringBuilder());
        return res;
    }

    private void backtraceForGenerateParenthesis(int left, int right, List<String> res, StringBuilder sb) {
        if (right < left || left < 0 || right < 0) {
            return;
        }
        if (left == 0 && right == 0) {
            res.add(sb.toString());
            return;
        }

        sb.append('(');
        backtraceForGenerateParenthesis(left - 1, right, res, sb);
        sb.deleteCharAt(sb.length() - 1);

        sb.append(')');
        backtraceForGenerateParenthesis(left, right - 1, res, sb);
        sb.deleteCharAt(sb.length() - 1);
    }

    //[23].合并K个升序链表
    public ListNode mergeKLists(ListNode[] lists) {
        //小顶堆
        Queue<ListNode> queue = new PriorityQueue<>((a, b) -> a.val - b.val);
        for (ListNode node : lists) {
            if (node != null) {
                queue.offer(node);
            }
        }
        ListNode dummyHead = new ListNode(-1), h = dummyHead;
        while (!queue.isEmpty()) {
            ListNode cur = queue.poll();
            h.next = cur;

            if (cur.next != null) {
                queue.offer(cur.next);
            }
            h = h.next;
        }
        return dummyHead.next;
    }

    //[24].两两交换链表中的节点
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode first = head, second = head.next;
        ListNode next = second.next;

        second.next = first;
        first.next = swapPairs(next);
        return second;
    }

    //[25].K个一组翻转链表
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null) return null;
        ListNode p = head;
        for (int i = 0; i < k; i++) {
            if (p == null) return head;
            p = p.next;
        }
        ListNode newHead = reverse(head, p);
        ListNode last = reverseKGroup(p.next, k);
        head.next = last;
        return newHead;
    }

    private ListNode reverse(ListNode head, ListNode end) {
        ListNode dummy = new ListNode(-1);
        ListNode p = head;
        while (p != end) {
            ListNode next = p.next;
            p.next = dummy.next;
            dummy.next = p;
            p = next;
        }

        end.next = dummy.next;
        dummy.next = end;
        return dummy.next;
    }

    //[26].删除有序数组中的重复项
    public int removeDuplicates(int[] nums) {
        if (nums.length == 0) return 0;
        //刚开始在一起， 当s与f的值不相等的时候，nums[++s] = nums[fast]
        int slow = 0, fast = 0;
        while (fast < nums.length) {
            if (nums[slow] != nums[fast]) {
                nums[++slow] = nums[fast];
            }
            fast++;
        }
        return slow + 1;
    }

    //[27].移除元素
    public int removeElement(int[] nums, int val) {
        //等于val的时候，快针+1，不等于的时候，快针复制到慢针，并且快慢针都+1
        int slow = 0, fast = 0;
        while (fast < nums.length) {
            if (nums[fast] != val) {
                nums[slow] = nums[fast];
                slow++;
                fast++;
            } else {
                fast++;
            }
        }
        return slow;
    }

    //[28].实现 strStr()
    public int strStr(String haystack, String needle) {
        if (needle == null || needle.length() == 0) return 0;
        if (haystack == null || haystack.length() == 0 || haystack.length() < needle.length())
            return -1;

        //abcdabcdef   abcdef
        int l = 0, r = 0;
        while (l < haystack.length() && r < needle.length()) {
            if (haystack.charAt(l) == needle.charAt(r)) {
                l++;
                r++;
            } else {
                //l的下一个位置重新开始
                l = l - r + 1;
                r = 0;
            }
        }
        return r == needle.length() ? l - needle.length() : -1;
    }

    //[31].下一个排列
    public void nextPermutation(int[] nums) {
        // 1 7 6 2 4 5
        // 1 7 6 2 5 4 3
        //后往前一直找到第一个 前面<后面 的值就是待交换的较小值
        //后往前找到第一个比较小值大的值
        //交换之后，从较小值后面全部翻转，因为都是递增排序，所以只需要全部翻转就可以。
        int n = nums.length;
        int i = n - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) i--;

        //降序，那么翻转全部
        if (i >= 0) {
            int j = n - 1;
            while (j > i && nums[j] <= nums[i]) j--;
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
        }
        reverse(nums, i + 1);
    }

    private void reverse(int[] nums, int start) {
        int end = nums.length - 1;
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }

    //[33].搜索旋转排序数组
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            }

            //说明left到mid是严格递增
            if (nums[mid] > nums[right]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    //mid肯定不等于target
                    left = mid + 1;
                }
            } else {
                //说明mid到right是严格递增
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    //mid肯定不等于target
                    right = mid - 1;
                }
            }
        }
        return -1;
    }

    //[34].在排序数组中查找元素的第一个和最后一个位置
    public int[] searchRange(int[] nums, int target) {
        int leftBound = findIndex(nums, target, true);
        int rightBound = findIndex(nums, target, false);
        return new int[]{leftBound, rightBound};
    }

    private int findIndex(int[] nums, int target, boolean isLeft) {
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
        if (isLeft) {
            if (left >= nums.length || nums[left] != target) return -1;
            return left;
        } else {
            if (right < 0 || nums[right] != target) return -1;
            return right;
        }
    }

    private int findLeftIndex(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (target == nums[mid]) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else if (target < nums[mid]) {
                right = mid - 1;
            }
        }

        if (left >= nums.length || nums[left] != target) return -1;
        return left;
    }

    private int findRightIndex(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (target == nums[mid]) {
                left = mid + 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else if (target < nums[mid]) {
                right = mid - 1;
            }
        }

        if (right < 0 || nums[right] != target) return -1;
        return right;
    }

    //版本2
    private int findLeftIndex2(int[] nums, int target) {
        //左边界，大于等于target
        //对于不存在的数而言，找的是大于它的第一个数
        //对于存在的多个数而言，找的是等于它的第一个数
        int left = 0, right = nums.length - 1;
        while (left < right) {
            //mid偏左
            int mid = left + (right - left) / 2;
            if (nums[mid] >= target) {
                //大于等于，可能是该值
                right = mid;
            } else {
                //值在右边，排除左边界
                left = mid + 1;
            }
        }
//        return nums[left] == target ? left : -1;
        return left;
    }

    //版本2
    private int findRightIndexV2(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        //右边界，小于等于target
        //对于不存在的数而言，找的是小于它的最后一个数
        //对于存在的多个数而言，找的是等于它的最后一个数
        while (left < right) {
            //mid偏右
            int mid = left + (right - left + 1) / 2;
            if (nums[mid] <= target) {
                //小于等于，可能是该值
                left = mid;
            } else {
                //值在左边，排除右边界
                right = mid - 1;
            }
        }
//        return nums[left] == target ? left : -1;
        return left;
    }

    //这确实是个右边界算法
    private int findRightIndexV3(int[] nums, int target) {
        //对于不存在和存在的数而言，找的是大于它的第一个数
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] <= target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    //[35].搜索插入位置
    public int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }

    //[36].有效的数独
    public boolean isValidSudoku(char[][] board) {
        //判断是否有重复值，可以用hash，减少时间复杂度
        for (int i = 0; i < 9; i++) {
            //数字是1-9
            int[] rowCount = new int[10];
            int[] colCount = new int[10];
            int[] areaCount = new int[10];
            for (int j = 0; j < 9; j++) {
                char ch = board[i][j];
                //i作为行，j作为列
                if (ch != '.') {
                    if (rowCount[ch - '0'] > 0) {
                        return false;
                    } else {
                        rowCount[ch - '0']++;
                    }
                }
                //i作为列， j作为行
                ch = board[j][i];
                if (ch != '.') {
                    if (colCount[ch - '0'] > 0) {
                        return false;
                    } else {
                        colCount[ch - '0']++;
                    }
                }
                //i作为格子序数， j格子内的索引
                ch = board[(i / 3) * 3 + j / 3][(i % 3) * 3 + j % 3];
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

    //[37].解数独
    public void solveSudoku(char[][] board) {
        backtraceSolveSudoku(board, 0, 0);
    }

    private boolean backtraceSolveSudoku(char[][] board, int x, int y) {
        if (y == 9) {
            return backtraceSolveSudoku(board, x + 1, 0);
        }

        //已经第10行了
        if (x == 9) {
            return true;
        }

        if (board[x][y] != '.') {
            return backtraceSolveSudoku(board, x, y + 1);
        }
        for (char i = '1'; i <= '9'; i++) {
            if (!isValidSudoku(board, x, y, i)) {
                continue;
            }
            board[x][y] = i;
            if (backtraceSolveSudoku(board, x, y + 1)) return true;
            board[x][y] = '.';
        }
        return false;
    }

    private boolean isValidSudoku(char[][] board, int x, int y, char ch) {
        for (int i = 0; i < 9; i++) {
            if (board[x][i] == ch) return false;
            if (board[i][y] == ch) return false;
            if (board[(x / 3) * 3 + i / 3][(y / 3) * 3 + i % 3] == ch) return false;
        }
        return true;
    }

    //[38].外观数列
    public String countAndSay(int n) {
        if (n == 1) return "1";

        String pre = countAndSay(n - 1);
        int count = 1;
        StringBuilder sb = new StringBuilder();
        for (int i = 1; i < pre.length(); i++) {
            if (pre.charAt(i - 1) == pre.charAt(i)) {
                count++;
            } else {
                sb.append(count).append(pre.charAt(i - 1));
                count = 1;
            }
        }
        sb.append(count).append(pre.charAt(pre.length() - 1));
        return sb.toString();
    }

    //[39].组合总和
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (candidates.length == 0) return res;
        backtraceForCombinationSum(candidates, 0, target, res, new LinkedList<>());
        return res;
    }

    private void backtraceForCombinationSum(int[] candidates, int s, int target, List<List<Integer>> res, LinkedList<Integer> select) {
        if (target == 0) {
            res.add(new ArrayList<>(select));
            return;
        }

        for (int i = s; i < candidates.length; i++) {
            if (candidates[i] > target) {
                continue;
            }
            select.addLast(candidates[i]);
            backtraceForCombinationSum(candidates, i, target - candidates[i], res, select);
            select.removeLast();
        }
    }

    //[40]组合总和 II
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (candidates.length == 0) return res;
        Arrays.sort(candidates);
        backtraceForCombinationSum2(candidates, 0, target, res, new LinkedList<>());
        return res;
    }

    private void backtraceForCombinationSum2(int[] candidates, int s, int target, List<List<Integer>> res, LinkedList<Integer> select) {
        if (target == 0) {
            res.add(new ArrayList<>(select));
            return;
        }

        for (int i = s; i < candidates.length; i++) {
            //本层如果有重复的话，只选第一个，后面的直接跳过
            if (i > s && candidates[i] == candidates[i - 1]) {
                continue;
            }
            if (candidates[i] > target) {
                continue;
            }
            select.addLast(candidates[i]);
            //元素不能重复选择，只能从下一个选择
            backtraceForCombinationSum2(candidates, i + 1, target - candidates[i], res, select);
            select.removeLast();
        }
    }

    //[43].字符串相乘
    public String multiply(String num1, String num2) {
        if (num1 == null || num2 == null) return "0";
        int m = num1.length(), n = num2.length();
        //最多有m+n位
        int[] res = new int[m + n];
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int multiple = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
                //乘积+ 进位信息
                multiple += res[i + j + 1];
                //低位信息直接覆盖
                res[i + j + 1] = multiple % 10;
                //进位信息要叠加
                res[i + j] += multiple / 10;
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < res.length; i++) {
            if (sb.length() == 0 && res[i] == 0) {
                continue;
            }
            sb.append(res[i]);
        }
        return sb.length() == 0 ? "0" : sb.toString();
    }

    //[42].接雨水
    public int trap(int[] height) {
        //找某侧最近一个比其大的值，使用单调栈维持栈内元素递减；
        //找某侧最近一个比其小的值，使用单调栈维持栈内元素递增
        int res = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < height.length; i++) {
            //   |   |
            //   | | |
            //   l c r， 维护单调递减栈，当大元素进栈，会压栈，弹出小的元素，那么此时栈顶一定大于弹出的元素
            while (!stack.isEmpty() && height[i] > height[stack.peek()]) {
                int cur = stack.pop();
                // 如果栈内没有元素，说明当前位置左边没有比其高的柱子，跳过
                if (stack.isEmpty()) {
                    continue;
                }
                // 左右位置，并有左右位置得出「宽度」和「高度」
                int l = stack.peek(), r = i;
                int w = r - l + 1 - 2;
                int h = Math.min(height[l], height[r]) - height[cur];
                res += w * h;
            }
            stack.push(i);
        }
        return res;
    }

    //[45].跳跃游戏 II
    public int jump(int[] nums) {
        int n = nums.length;
        //刚开始为-1，防止多计数
        int res = -1, nextMaxIndex = 0, end = 0;
        for (int i = 0; i < n; i++) {
            nextMaxIndex = Math.max(i + nums[i], nextMaxIndex);
            //到达最远地方
            if (i == end) {
                //更新最远距离
                end = nextMaxIndex;
                //计数
                res++;
            }
        }
        return res;
    }

    //[46].全排列
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        LinkedList<Integer> path = new LinkedList<>();
        dfsForPermute(res, path, nums);
        return res;
    }

    private void dfsForPermute(List<List<Integer>> res, LinkedList<Integer> path, int[] nums) {
        if (path.size() == nums.length) {
            res.add(new LinkedList<>(path));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            if (path.contains(nums[i])) {
                continue;
            }
            path.addLast(nums[i]);
            dfsForPermute(res, path, nums);
            path.removeLast();
        }
    }

    //[47].全排列 II
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length == 0) {
            return res;
        }
        Arrays.sort(nums);
        LinkedList<Integer> path = new LinkedList<>();
        boolean[] visited = new boolean[nums.length];
        dfsForPermuteUnique(res, path, nums, visited);
        return res;
    }

    private void dfsForPermuteUnique(List<List<Integer>> res, LinkedList<Integer> path, int[] nums, boolean[] visited) {
        if (path.size() == nums.length) {
            res.add(new LinkedList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i]) {
                continue;
            }
            //决策树画完之后，发现01这种状态需要剪枝，意思是重复的数。
            //一定从左边往右边选: 如果左边的还没有选，则右边的也不选，直接跳过。
            if (i > 0 && nums[i] == nums[i - 1] && !visited[i - 1]) {
                continue;
            }
            visited[i] = true;
            path.add(nums[i]);
            dfsForPermuteUnique(res, path, nums, visited);
            path.removeLast();
            visited[i] = false;
        }
    }

    //[48].旋转图像
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        //对角线翻转
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }

        //左右互换
        for (int i = 0; i < n; i++) {
            int left = 0, right = n - 1;
            while (left < right) {
                int temp = matrix[i][left];
                matrix[i][left] = matrix[i][right];
                matrix[i][right] = temp;
                left++;
                right--;
            }
        }
    }

    //[49].字母异位词分组
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> group = new HashMap<>();

        for (String str : strs) {
            char[] array = str.toCharArray();
            Arrays.sort(array);
            String key = String.valueOf(array);
            List<String> list = group.getOrDefault(key, new ArrayList<>());
            list.add(str);

            group.put(key, list);
        }
        List<List<String>> res = new ArrayList<>();

        for (List<String> item : group.values()) {
            res.add(item);
        }
        return res;
    }

    //[50].Pow(x, n)
    public double myPow(double x, int n) {
        if (n == 0) return 1;
        if (n == 1) return x;
        if (n == -1) return 1 / x;

        double half = myPow(x, n / 2);
        double left = myPow(x, n % 2);
        return half * half * left;
    }

    //[51].N皇后
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            sb.append('.');
        }
        LinkedList select = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            select.add(sb.toString());
        }
        backtraceForSolveNQueens(n, 0, res, select);
        return res;
    }

    private void backtraceForSolveNQueens(int n, int row, List<List<String>> res, LinkedList<String> select) {
        if (row == n) {
            res.add(new ArrayList<>(select));
            return;
        }
        char[] arr = select.get(row).toCharArray();
        for (int col = 0; col < n; col++) {
            if (!isValid(select, row, col)) {
                continue;
            }
            arr[col] = 'Q';
            select.set(row, new String(arr));

            backtraceForSolveNQueens(n, row + 1, res, select);

            arr[col] = '.';
            select.set(row, new String(arr));
        }
    }

    private boolean isValid(LinkedList<String> select, int row, int col) {
        int n = select.get(row).length();
        //每次行上是一个选择，所以不需要判断行是否合法，因为肯定合法
        //列不合法
        for (int i = 0; i < n; i++) {
            if (select.get(i).charAt(col) == 'Q') {
                return false;
            }
        }
        //左上不合法
        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
            if (select.get(i).charAt(j) == 'Q') {
                return false;
            }
        }

        //右上不合法
        for (int i = row, j = col; i >= 0 && j < n; i--, j++) {
            if (select.get(i).charAt(j) == 'Q') {
                return false;
            }
        }
        return true;
    }

    //[53].最大子数组和
    public int maxSubArray(int[] nums) {
        int n = nums.length;
        //以i为结尾的最大连续子数组和，当dp[i-1] + nums[i] < nums[i]的时候，dp[i] = nums[i]
        int res = Integer.MIN_VALUE;
        int[] dp = new int[n];
        dp[0] = nums[0];
        for (int i = 1; i < n; i++) {
            if (dp[i - 1] + nums[i] < nums[i]) {
                dp[i] = nums[i];
            } else {
                dp[i] = dp[i - 1] + nums[i];
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    //[54].螺旋矩阵
    public List<Integer> spiralOrder(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int top = 0, bottom = m - 1;
        int left = 0, right = n - 1;
        List<Integer> res = new ArrayList<>();
        while (res.size() < m * n) {
            if (top <= bottom) {
                for (int i = left; i <= right; i++) {
                    res.add(matrix[top][i]);
                }
                top++;
            }
            if (left <= right) {
                for (int i = top; i <= bottom; i++) {
                    res.add(matrix[i][right]);
                }
                right--;
            }

            if (top <= bottom) {
                for (int i = right; i >= left; i--) {
                    res.add(matrix[bottom][i]);
                }
                bottom--;
            }

            if (left <= right) {
                for (int i = bottom; i >= top; i--) {
                    res.add(matrix[i][left]);
                }
                left++;
            }
        }
        return res;
    }

    //[55].跳跃游戏
    public boolean canJump(int[] nums) {
        int n = nums.length;
        int maxIndex = 0;
        //最后一个需要排除掉，因为可能正好相等
        for (int i = 0; i < n - 1; i++) {
            maxIndex = Math.max(maxIndex, nums[i] + i);
            if (maxIndex <= i) {
                return false;
            }
        }
        return true;
    }

    //[56].合并区间
    public int[][] merge(int[][] intervals) {
        //1 2  1 5  3 6  7 9
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        LinkedList<int[]> res = new LinkedList<>();
        res.addLast(intervals[0]);
        for (int i = 1; i < intervals.length; i++) {
            int[] last = res.getLast();
            if (intervals[i][0] <= last[1]) {
                last[1] = Math.max(last[1], intervals[i][1]);
            } else {
                res.addLast(intervals[i]);
            }
        }
        return res.toArray(new int[res.size()][]);
    }

    //[57].插入区间
    public int[][] insert(int[][] intervals, int[] newInterval) {
        if (newInterval.length == 0) return intervals;
        LinkedList<int[]> res = new LinkedList<>();
        int left = newInterval[0], right = newInterval[1];
        for (int i = 0; i < intervals.length; i++) {
            int[] interval = intervals[i];
            if (interval[0] > right) {
                res.add(new int[]{left, right});
                //在插入区间右侧，且不相交，则添加插入区域
                //同时更新插入区域，保证下一次添加为插入区域
                left = interval[0];
                right = interval[1];
            } else if (left > interval[1]) {
                //在插入区间左侧，且不相交，则添加左侧区域
                res.add(interval);
            } else {
                //相交，维护区域
                left = Math.min(interval[0], left);
                right = Math.max(interval[1], right);
            }
        }
        //还剩最后一个区域需要添加
        res.add(new int[]{left, right});
        return res.toArray(new int[][]{});
    }

    //[58].最后一个单词的长度
    public int lengthOfLastWord(String s) {
        int end = s.length() - 1;
        while (end >= 0 && s.charAt(end) == ' ') end--;
        int start = end;
        while (start >= 0 && s.charAt(start) != ' ') start--;
        return end - start;
    }

    //[59].螺旋矩阵 II
    public int[][] generateMatrix(int n) {
        int top = 0, bottom = n - 1;
        int left = 0, right = n - 1;
        int num = 0;
        int[][] res = new int[n][n];
        //每个都从1开始计数，那么当num == n平方就是结束条件，能够正常退出
        while (num < n * n) {
            if (top <= bottom) {
                for (int i = left; i <= right; i++) {
                    res[top][i] = ++num;
                }
                top++;
            }

            if (left <= right) {
                for (int i = top; i <= bottom; i++) {
                    res[i][right] = ++num;
                }
                right--;
            }

            if (top <= bottom) {
                for (int i = right; i >= left; i--) {
                    res[bottom][i] = ++num;
                }
                bottom--;
            }

            if (left <= right) {
                for (int i = bottom; i >= top; i--) {
                    res[i][left] = ++num;
                }
                left++;
            }
        }
        return res;
    }

    //[61].旋转链表
    public ListNode rotateRight(ListNode head, int k) {
        int count = 0;
        ListNode p = head, last = null;
        while (p != null) {
            last = p;
            p = p.next;
            count++;
        }
        int realK;
        if (count == 0 || count == 1 || (realK = k % count) == 0) return head;

        ListNode slow = head, fast = head;
        for (int i = 0; i < realK; i++) {
            fast = fast.next;
        }
        ListNode pre = null;
        while (fast != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next;
        }
        pre.next = null;
        last.next = head;
        return slow;
    }

    //[62].不同路径
    public int uniquePaths(int m, int n) {
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        //          (i-1,j)
        //             ↓
        //(i,j-1) →  (i,j)
        //遍历顺序是从左往右，垂直投影，砍掉i维度之后，dp[i-1][j]的值就是之前的dp[j]
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[j] = dp[j] + dp[j - 1];
            }
        }
        return dp[n - 1];
    }

    //[63].不同路径 II
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length, n = obstacleGrid[0].length;
        if (m == 0 || n == 0) return 0;

        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            if (obstacleGrid[i][0] == 0) {
                dp[i][0] = 1;
            } else {
                break;
            }
        }
        for (int i = 0; i < n; i++) {
            if (obstacleGrid[0][i] == 0) {
                dp[0][i] = 1;
            } else {
                break;
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    //[64].最小路径和
    public int minPathSum(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        //从[0][0]到[i][j]的最小路径和
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < m; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int j = 1; j < n; j++) {
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.min(dp[i][j - 1], dp[i - 1][j]) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }

    //[66].加一
    public int[] plusOne(int[] digits) {
        int capacity = 1;
        int n = digits.length;
        for (int i = n - 1; i >= 0; i--) {
            int sum = capacity + digits[i];
            capacity = sum / 10;
            digits[i] = sum % 10;
            //进位只可能是10，所以不是0则没有进位，直接退出
            if (digits[i] != 0) {
                return digits;
            }
        }
        //[9,9,9] => [1,0,0,0]
        digits = new int[n + 1];
        digits[0] = 1;
        return digits;
    }

    //[67].二进制求和
    public String addBinary(String a, String b) {
        int aLen = a.length(), bLen = b.length();
        int i = aLen - 1, j = bLen - 1;
        StringBuilder sb = new StringBuilder();
        int cap = 0;
        while (i >= 0 || j >= 0 || cap != 0) {
            int first = i >= 0 ? a.charAt(i--) - '0' : 0;
            int second = j >= 0 ? b.charAt(j--) - '0' : 0;
            int sum = first + second + cap;
            sb.insert(0, sum % 2);

            cap = sum / 2;
        }
        return sb.toString();
    }

    //[69].Sqrt(x)
    public int mySqrt(int x) {
        if (x == 0) return 0;
        int left = 1, right = x;
        while (left < right) {
            int mid = left + (right - left + 1) / 2;
            if (mid == x / mid) {
                left = mid;
            } else if (mid < x / mid) {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }

    //[70].爬楼梯
    public int climbStairs(int n) {
        if (n == 0) return 0;
        //爬到第i阶，有多少种方案
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 2] + dp[i - 1];
        }
        return dp[n];
    }

    //[71].简化路径
    public String simplifyPath(String path) {
        String[] strings = path.split("/");
        Stack<String> stack = new Stack<>();
        for (String str : strings) {
            if (!stack.isEmpty() && str.equals("..")) {
                stack.pop();
            } else if (!str.equals("") && !str.equals(".") && !str.equals("..")) {
                //两个//中间就是空串
                stack.push(str);
            }
        }
        if (stack.isEmpty()) return "/";

        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            sb.insert(0, "/" + stack.pop());
        }
        return sb.toString();
    }

    //[73].矩阵置零
    public void setZeroes(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        boolean rowZero = false, colZero = false;
        for (int i = 1; i < m; i++) {
            if (matrix[i][0] == 0) {
                colZero = true;
                break;
            }
        }
        for (int j = 1; j < n; j++) {
            if (matrix[0][j] == 0) {
                rowZero = true;
                break;
            }
        }
        if (matrix[0][0] == 0) {
            colZero = true;
            rowZero = true;
        }
        //映射到第一行和第一列上
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        //根据第一行和第一列上的修改为0
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        //之前第一列有0，只需要设置第一列
        if (colZero) {
            for (int i = 0; i < m; i++) {
                matrix[i][0] = 0;
            }
        }
        //之前第一行有0，只需要设置第一行
        if (rowZero) {
            for (int i = 0; i < n; i++) {
                matrix[0][i] = 0;
            }
        }
    }

    //[74].搜索二维矩阵
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int left = 0, right = m * n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int x = mid / m;
            int y = mid % m;
            if (matrix[x][y] == target) {
                return true;
            } else if (matrix[x][y] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return false;
    }

    //[75].颜色分类
    public void sortColors(int[] nums) {
        //left为0存放的下一个位置，right为2存放的下一个位置
        int left = 0, right = nums.length - 1;
        int i = 0;
        //只要遍历到2的位置即可，right可能是2，也可能不是2
        while (i <= right) {
            if (nums[i] == 0) {
                //left控制最后一个0的位置，左边不能出现2
                //i推进
                swap(nums, i++, left++);
            } else if (nums[i] == 1) {
                i++;
            } else {
                //重新交换到i位置上的可能是0,1,2,如果是2，继续交换到right上。
                swap(nums, i, right--);
            }
        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    //[77].组合
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        backtraceForCombine(n, k, 1, res, new LinkedList<>());
        return res;
    }

    private void backtraceForCombine(int n, int k, int s, List<List<Integer>> res, LinkedList<Integer> select) {
        if (select.size() == k) {
            res.add(new ArrayList<>(select));
            return;
        }

        for (int i = s; i <= n; i++) {
            select.addLast(i);
            backtraceForCombine(n, k, i + 1, res, select);
            select.removeLast();
        }
    }

    //[78].子集
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        backtraceForSubsets(nums, 0, res, new LinkedList<>());
        return res;
    }

    private void backtraceForSubsets(int[] nums, int s, List<List<Integer>> res, LinkedList<Integer> select) {
        //1 2 3
        res.add(new ArrayList<>(select));
        for (int i = s; i < nums.length; i++) {
            select.addLast(nums[i]);
            backtraceForSubsets(nums, i + 1, res, select);
            select.removeLast();
        }
    }

    //[79].单词搜索
    public boolean exist(char[][] board, String word) {
        int m = board.length;
        int n = board[0].length;
        boolean[][] visit = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (backtraceForExist(board, i, j, word, 0, visit)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean backtraceForExist(char[][] board, int x, int y, String word, int s, boolean[][] visit) {
        if (board[x][y] != word.charAt(s)) {
            return false;
        }
        if (s == word.length() - 1) {
            return true;
        }

        visit[x][y] = true;
        int[][] direct = new int[][]{{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
        int m = board.length;
        int n = board[0].length;
        for (int[] di : direct) {
            int newX = x + di[0];
            int newY = y + di[1];
            if (newX >= 0 && newX < m
                    && newY >= 0 && newY < n
                    && !visit[newX][newY]
                    && backtraceForExist(board, newX, newY, word, s + 1, visit)) {
                return true;
            }
        }

        visit[x][y] = false;
        return false;
    }

    //[80].删除有序数组中的重复项 II
    public int removeDuplicates2(int[] nums) {
        int n = nums.length;
        if (n <= 2) return n;
//        int slow = 2, fast = 2;
//        while (fast < nums.length) {
//            if (nums[slow - 2] != nums[fast]) {
//                nums[slow++] = nums[fast];
//            }
//            fast++;
//        }
//        return slow;
        int k = 2;
        int idx = 0;
        for (int num : nums) {
            if (idx < k || nums[idx - k] != num) nums[idx++] = num;
        }
        return idx;
    }

    //[81].搜索旋转排序数组 II
    public boolean search3(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] == target) {
                return true;
            }

            //10111 11101 这两种情况中，没办法判断递增区间走向，所以砍掉左边的相同的
            if (nums[mid] == nums[left]) {
                left++;
            } else if (nums[mid] > nums[right]) {
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
        return false;
    }

    //[82].删除排序链表中的重复元素 II
    public ListNode deleteDuplicates2(ListNode head) {
        ListNode dummy = new ListNode(-1);
        //建一个「虚拟头节点」dummy 以减少边界判断，往后的答案链表会接在 dummy 后面
        //使用 tail 代表当前有效链表的结尾
        //通过原输入的 head 指针进行链表扫描
        ListNode tail = dummy;
        while (head != null) {
            if (head.next == null || head.val != head.next.val) {
                tail.next = head;
                tail = tail.next;
            }
            //如果head与下一节点值相同个，跳过相同节点
            while (head.next != null && head.val == head.next.val) {
                head = head.next;
            }
            //后一个节点就是需要的
            head = head.next;
        }
        //把不合法的删掉
        tail.next = null;
        return dummy.next;
    }

    //[83].删除排序链表中的重复元素
    public ListNode deleteDuplicates(ListNode head) {
        ListNode slow = head, fast = head;

        while (fast != null) {
            if (slow.val != fast.val) {
                slow.next = fast;
                slow = slow.next;
            }
            fast = fast.next;
        }
        slow.next = null;
        return head;
    }

    //[86].分隔链表
    public ListNode partition(ListNode head, int x) {
        ListNode sDummy = new ListNode(-1), fDummy = new ListNode(-1), cur = head, s = sDummy, f = fDummy;
        while (cur != null) {
            ListNode next = cur.next;
            if (cur.val < x) {
                s.next = cur;
                s = s.next;
            } else {
                f.next = cur;
                f = f.next;
            }
            //每次都断掉防止麻烦
            cur.next = null;
            cur = next;
        }
        s.next = fDummy.next;
        return sDummy.next;
    }

    //[88].合并两个有序数组
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        //因为数据是最后是往nums1中写入的，而且是从小到大，为了避免前面交互，所有从后往前遍历
        int i = m - 1, j = n - 1, index = nums1.length - 1;
        //我也可以先求公共的，然后再求剩下的
        //或者通过判断条件，特殊判断来统一
        while (i >= 0 || j >= 0) {
            if (i == -1) {
                nums1[index--] = nums2[j--];
            } else if (j == -1) {
                nums1[index--] = nums1[i--];
            } else if (nums1[i] > nums2[j]) {
                nums1[index--] = nums1[i--];
            } else {
                nums1[index--] = nums2[j--];
            }
        }
    }

    //[89].格雷编码
    public List<Integer> grayCode(int n) {
        boolean[] visit = new boolean[1 << n];
        LinkedList<Integer> select = new LinkedList<>();
        backtraceForGrayCode(n, 0, visit, select);
        return select;
    }

    private boolean backtraceForGrayCode(int n, int cur, boolean[] visit, LinkedList<Integer> select) {
        if (select.size() == 1 << n) {
            return true;
        }

        select.add(cur);
        visit[cur] = true;
        for (int i = 0; i < n; i++) {
            //异或保证有一位不相同其他都相同
            int next = cur ^ 1 << i;
            if (!visit[next] && backtraceForGrayCode(n, next, visit, select)) {
                return true;
            }
        }
        //肯定有唯一解，不需要撤销
        return false;
    }

    //[90].子集 II
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        backtraceForSubsetsWithDup(nums, 0, res, new LinkedList<>());
        return res;
    }

    private void backtraceForSubsetsWithDup(int[] nums, int s, List<List<Integer>> res, LinkedList<Integer> select) {
        res.add(new ArrayList<>(select));
        for (int i = s; i < nums.length; i++) {
            //本层中有重复的，则跳过，永远直选第一个
            if (i > s && nums[i] == nums[i - 1]) {
                continue;
            }

            select.addLast(nums[i]);
            backtraceForSubsetsWithDup(nums, i + 1, res, select);
            select.removeLast();
        }
    }


    //[92].反转链表 II
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode pre = dummy;
        for (int i = 0; i < left - 1; i++) {
            pre = pre.next;
        }
        ListNode cur = pre.next;
        ListNode next;
        for (int i = 0; i < right - left; i++) {
            next = cur.next;
            //移动的永远是cur
            cur.next = next.next;
            //此处的cur是会变动的，所以接的是pre.next节点
            next.next = pre.next;
            pre.next = next;
        }
        return dummy.next;
    }

    //[93].复原 IP 地址
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        dfsForRestoreIpAddresses(s, 0, new LinkedList<>(), res);
        return res;
    }

    private void dfsForRestoreIpAddresses(String s, int start, LinkedList<String> select, List<String> res) {
        if (select.size() > 4) {
            return;
        }
        if (select.size() >= 4 && start != s.length()) {
            return;
        }
        if (start == s.length() && select.size() == 4) {
            res.add(String.join(".", select));
            return;
        }

        for (int i = start; i < s.length(); i++) {
            String choice = s.substring(start, i + 1);
            if (choice.length() > 3 || choice.startsWith("0") && choice.length() > 1 || Integer.parseInt(choice) > 255) {
                return;
            }
            select.addLast(choice);
            dfsForRestoreIpAddresses(s, i + 1, select, res);
            select.removeLast();
        }
    }

    //[94].二叉树的中序遍历
    public List<Integer> inorderTraversalV2(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        List<Integer> res = new ArrayList<>();
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

    //[95].不同的二叉搜索树 II
    public List<TreeNode> generateTrees(int n) {
        if (n == 0) return null;
        return dfsForGenerateTrees(1, n);
    }

    private List<TreeNode> dfsForGenerateTrees(int start, int end) {
        List<TreeNode> res = new ArrayList<>();
        if (start > end) {
            res.add(null);
            return res;
        }
        for (int i = start; i <= end; i++) {
            List<TreeNode> leftTree = dfsForGenerateTrees(start, i - 1);
            List<TreeNode> rightTree = dfsForGenerateTrees(i + 1, end);
            //始终都只有一个节点
            for (TreeNode left : leftTree) {
                for (TreeNode right : rightTree) {
                    TreeNode root = new TreeNode(i);
                    root.left = left;
                    root.right = right;
                    res.add(root);
                }
            }
        }
        return res;
    }

    //[96].不同的二叉搜索树
    public int numTrees(int n) {
        //方法1 递归解决
        //return dfsForNumTrees(new int[n + 1][n + 1], 1, n);

        //方法2
        // 1  1
        // 2  2
        // 3  2 + 1 + 2
        // 4  5 + 1 * 2 + 2*1 + 5
        //i个节点的数量，而不是前i个节点的数量
        //dp[i] = dp[0] * dp [i - 1] + dp[1] * dp[i-2] + ... + dp[i-1]* dp[0]
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

    private int dfsForNumTrees(int[][] memo, int start, int end) {
        if (start > end) {
            return 1;
        }
        if (memo[start][end] != 0) {
            return memo[start][end];
        }
        int res = 0;
        for (int i = start; i <= end; i++) {
            int left = dfsForNumTrees(memo, start, i - 1);
            int right = dfsForNumTrees(memo, i + 1, end);
            res += left * right;
        }
        memo[start][end] = res;
        return res;
    }

    //[97].交错字符串
    public boolean isInterleave(String s1, String s2, String s3) {
        int m = s1.length(), n = s2.length();
        //aa bb  abba ab已经被匹配成功，abb是继续选择s2字符，也是交错字符串的一种场景，要么选择s1，要么继续选择s2。
        //s1长度为i和s2长度为j 与s3[0...i+j-1] 能否组成交替字符串
        boolean[][] dp = new boolean[m + 1][n + 1];
        //空串可以构成空串
        dp[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                //i这边表示长度，索引为i-1，s3的索引为i+j-1
                if (i > 0 && dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1)) {
                    dp[i][j] = true;
                }
                if (j > 0 && dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1)) {
                    dp[i][j] = true;
                }
            }
        }
        return dp[m][n];
    }

    //[98].验证二叉搜索树
    public boolean isValidBST(TreeNode root) {
        return dfsIsValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    private boolean dfsIsValidBST(TreeNode root, Long min, Long max) {
        if (root == null) return true;
        if (min < root.val && root.val < max) {
            return dfsIsValidBST(root.left, min, (long) root.val) &&
                    dfsIsValidBST(root.right, (long) root.val, max);
        }
        return false;
    }

    //[99].恢复二叉搜索树
    private TreeNode preNode, firstMax, lastMin;

    public void recoverTree(TreeNode root) {
        dfsForRecoverTree(root);
        if (firstMax != null && lastMin != null) {
            int temp = firstMax.val;
            firstMax.val = lastMin.val;
            lastMin.val = temp;
        }
    }

    private void dfsForRecoverTree(TreeNode root) {
        if (root == null) return;
        dfsForRecoverTree(root.left);

        //中序遍历性质应用
        //左右子树交换会有两段，1 2 8 4 5 6 7 3
        //第一次前节点大于后节点，第一次前节点是交换节点，最后一次的前节点大于后节点，最后一次后节点是交换节点
        if (preNode != null && preNode.val > root.val) {
            //第一次赋值
            if (firstMax == null) firstMax = preNode;
            //每次赋值，直到最后一次后节点
            lastMin = root;
        }
        preNode = root;
        dfsForRecoverTree(root.right);
    }

    //[100].相同的树
    public boolean isSameTree(TreeNode p, TreeNode q) {
        return dfsIsSameTree(p, q);
    }

    private boolean dfsIsSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null && q != null) return false;
        if (p != null && q == null) return false;
        if (p.val != q.val) return false;
        return dfsIsSameTree(p.left, q.left) && dfsIsSameTree(p.right, q.right);
    }

    //[101].对称二叉树
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return dfsIsSymmetric(root.left, root.right);
    }

    private boolean dfsIsSymmetric(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        //是否为镜像，意味着，主节点相等， p的左节点 = q的右节点， p的右节点 = q的左节点
        if (p == null && q != null) return false;
        if (p != null && q == null) return false;
        if (p.val != q.val) return false;
        return dfsIsSymmetric(p.left, q.right) && dfsIsSymmetric(p.right, q.left);
    }

    //[102].二叉树的层序遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) return null;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        List<List<Integer>> res = new ArrayList<>();
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
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        if (root == null) return null;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean flag = true;
        List<List<Integer>> res = new ArrayList<>();
        while (!queue.isEmpty()) {
            LinkedList<Integer> list = new LinkedList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();
                if (flag) {
                    list.addLast(cur.val);
                } else {
                    list.addFirst(cur.val);
                }

                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }
            }
            res.add(list);
            flag = !flag;
        }
        return res;
    }

    //[104].二叉树的最大深度
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    //[105].从前序与中序遍历序列构造二叉树
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int pLen = preorder.length;
        int iLen = inorder.length;
        if (pLen != iLen) {
            return null;
        }
        Map<Integer, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < iLen; i++) {
            indexMap.put(inorder[i], i);
        }
        return dfsForBuildTree(preorder, inorder, 0, pLen - 1, 0, iLen - 1, indexMap);
    }

    private TreeNode dfsForBuildTree(int[] preorder, int[] inorder, int ps, int pe, int is, int ie, Map<Integer, Integer> indexMap) {
        if (ps > pe || is > ie) return null;
        int rootVal = preorder[ps];
        TreeNode root = new TreeNode(rootVal);
        int index = indexMap.get(rootVal);
        TreeNode left = dfsForBuildTree(preorder, inorder, ps + 1, index - is + ps, is, index - 1, indexMap);
        TreeNode right = dfsForBuildTree(preorder, inorder, index - is + ps + 1, pe, index + 1, ie, indexMap);
        root.left = left;
        root.right = right;
        return root;
    }

    //[106].从中序与后序遍历序列构造二叉树
    public TreeNode buildTree2(int[] inorder, int[] postorder) {
        int iLen = inorder.length;
        int pLen = postorder.length;
        if (iLen != pLen) return null;
        Map<Integer, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < iLen; i++) {
            indexMap.put(inorder[i], i);
        }
        return dfsForBuildTree2(inorder, postorder, 0, iLen - 1, 0, pLen - 1, indexMap);
    }

    private TreeNode dfsForBuildTree2(int[] inorder, int[] postorder, int is, int ie, int ps, int pe, Map<Integer, Integer> indexMap) {
        if (is > ie || ps > pe) {
            return null;
        }
        int rootVal = postorder[pe];
        TreeNode root = new TreeNode(rootVal);
        int index = indexMap.get(rootVal);

        TreeNode left = dfsForBuildTree2(inorder, postorder, is, index - 1, ps, ps + index - is - 1, indexMap);
        TreeNode right = dfsForBuildTree2(inorder, postorder, index + 1, ie, ps + index - is, pe - 1, indexMap);
        root.left = left;
        root.right = right;
        return root;
    }

    //[107].二叉树的层序遍历 II
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> res = new LinkedList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if (root != null) {
            queue.offer(root);
        }
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
            res.add(0, level);
        }
        return res;
    }

    //[108].将有序数组转换为二叉搜索树
    public TreeNode sortedArrayToBST(int[] nums) {
        return dfsForSortedArrayToBST(nums, 0, nums.length - 1);
    }

    private TreeNode dfsForSortedArrayToBST(int[] nums, int s, int e) {
        if (s > e) return null;
        int mid = s + (e - s) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        TreeNode left = dfsForSortedArrayToBST(nums, s, mid - 1);
        TreeNode right = dfsForSortedArrayToBST(nums, mid + 1, e);
        root.left = left;
        root.right = right;
        return root;
    }

    //[109].有序链表转换二叉搜索树
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) return null;
        ListNode slow = head, fast = head, pre = null;
        while (fast != null && fast.next != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        TreeNode root = new TreeNode(slow.val);
        //只有一个节点了
        if (pre == null) return root;

        //断开前面一个节点
        pre.next = null;
        root.left = sortedListToBST(head);
        root.right = sortedListToBST(slow.next);
        return root;
    }

    //[110].平衡二叉树
    public boolean isBalanced(TreeNode root) {
        return dfsForIsBalanced(root) >= 0;
    }

    //不平衡就是-1， 否则返回高度
    private int dfsForIsBalanced(TreeNode root) {
        if (root == null) return 0;
        int left = dfsForIsBalanced(root.left);
        int right = dfsForIsBalanced(root.right);

        if (left == -1 || right == -1 || Math.abs(left - right) > 1) {
            return -1;
        }
        return Math.max(left, right) + 1;
    }

    //[111].二叉树的最小深度
    public int minDepth(TreeNode root) {
        if (root == null) return 0;
        int left = minDepth(root.left);
        int right = minDepth(root.right);
        //有一棵子树没有，此时高度应该是2， 而不是下面的1，计算就会不对。
        if (left == 0 || right == 0) {
            return left + right + 1;
        }
        return Math.min(left, right) + 1;
    }

    //[112].路径总和
    public boolean hasPathSum(TreeNode root, int targetSum) {
        //一定不是叶子节点
        if (root == null) return false;
        //判断叶子节点标准
        if (root.left == null && root.right == null) {
            return targetSum == root.val;
        }
        return hasPathSum(root.left, targetSum - root.val)
                || hasPathSum(root.right, targetSum - root.val);
    }

    //[113].路径总和 II
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> res = new ArrayList<>();
        dfsForPathSum(root, targetSum, new LinkedList<>(), res);
        return res;
    }

    private void dfsForPathSum(TreeNode root, int targetSum, LinkedList<Integer> path, List<List<Integer>> res) {
        if (root == null) return;

        //前序遍历
        path.addLast(root.val);
        if (root.left == null && root.right == null && targetSum == root.val) {
            res.add(new ArrayList<>(path));
            //回溯撤销节点的，加了return，会导致叶子节点会有撤销成功，导致路径上少减少一次撤销，从而使得下一次的选择会多一个节点。
            //主要取决于前序遍历顺序不能变更。
        }

        dfsForPathSum(root.left, targetSum - root.val, path, res);
        dfsForPathSum(root.right, targetSum - root.val, path, res);
        path.removeLast();
    }

    //[114].二叉树展开为链表
    public void flatten(TreeNode root) {
        if (root == null) return;

        flatten(root.left);
        flatten(root.right);

        TreeNode left = root.left;
        TreeNode right = root.right;

        root.right = left;
        root.left = null;

        TreeNode cur = root;
        //最后一个叶子节点
        while (cur.right != null) {
            cur = cur.right;
        }
        cur.right = right;
    }

    //[116].填充每个节点的下一个右侧节点指针
    public Node connect(Node root) {
        if (root == null) return null;
        dfsForConnect(root.left, root.right);
        return root;
    }

    private void dfsForConnect(Node left, Node right) {
        if (left == null || right == null) {
            return;
        }
        left.next = right;
        dfsForConnect(left.left, left.right);
        dfsForConnect(right.left, right.right);
        dfsForConnect(left.right, right.left);
    }

    //[117].填充每个节点的下一个右侧节点指针 II
    public Node connect2(Node root) {
//        if (root == null) return null;
//        if (root.left != null) {
//            if (root.right != null) {
//                root.left.next = root.right;
//            } else {
//                root.left.next = findFirstNext(root.next);
//            }
//        }
//        //老版本存在错误，就是右边没有链接，丢了信息，右边不为空时，应该要再去找有节点的下一个节点才行
//        if (root.right != null && root.next != null) {
//            root.right.next = findFirstNext(root.next);
//        }
//
//        //先链接右边，再链接左边，因为findFirstNext会提前将父节点的next信息维护好，不然left子节点没办法找到下一个链接的节点
//        connect2(root.right);
//        connect2(root.left);

        if (root == null) return null;
        //cur代表当前的头结点
        Node cur = root;
        while (cur != null) {
            //下一层的虚拟节点
            Node nextLayerDummy = new Node(0);
            Node pre = nextLayerDummy;
            //遍历这一层的每个节点
            while (cur != null) {
                if (cur.left != null) {
                    //上个节点链接左孩子节点
                    pre.next = cur.left;
                    pre = pre.next;
                }
                if (cur.right != null) {
                    pre.next = cur.right;
                    pre = pre.next;
                }
                cur = cur.next;
            }
            //这一层遍历完，到下一层的头结点
            cur = nextLayerDummy.next;
        }
        return root;
    }

    private Node findFirstNext(Node root) {
        Node next = root, nextRight = null;
        while (next != null) {
            if (next.left != null) {
                nextRight = next.left;
                break;
            }
            if (next.right != null) {
                nextRight = next.right;
                break;
            }
            next = next.right;
        }
        return nextRight;
    }

    //[118].杨辉三角
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            List<Integer> level = new ArrayList<>();
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) {
                    level.add(1);
                } else {
                    level.add(res.get(i - 1).get(j - 1) + res.get(i - 1).get(j));
                }
            }
            res.add(level);
        }
        return res;
    }

    //[119].杨辉三角 II
    public List<Integer> getRow(int rowIndex) {
        List<Integer> res = new ArrayList<>();
        res.add(1);
        for (int i = 1; i <= rowIndex; i++) {
            res.add(0);
            for (int j = i; j > 0; j--) {
                res.set(j, res.get(j) + res.get(j - 1));
            }
        }
        return res;
    }

    //[120].三角形最小路径和
    public int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size();
        //走到(i, j)点的最小路径和
        int[][] dp = new int[n][n];
        dp[0][0] = triangle.get(0).get(0);
        for (int i = 1; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (j == 0) {
                    dp[i][j] = dp[i - 1][j] + triangle.get(i).get(j);
                } else if (j == i) {
                    dp[i][j] = dp[i - 1][j - 1] + triangle.get(i).get(j);
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle.get(i).get(j);
                }
            }
        }
        int min = dp[n - 1][0];
        for (int i = 1; i < n; i++) {
            min = Math.min(min, dp[n - 1][i]);
        }
        return min;
    }

    //[120].三角形最小路径和（空间压缩）
    public int minimumTotal2(List<List<Integer>> triangle) {
        int n = triangle.size();
        //到底层i的最短路径和
        int[] dp = new int[n];
        dp[0] = triangle.get(0).get(0);
        int pre = 0, cur;
        //  pre          cur, pre'     cur'
        // (i-1, j-1)   (i-1, j)     (i-1, j+1)
        //        ＼        ↓    ＼      ↓
        //               (i, j)       (i, j+1)
        for (int i = 1; i < n; i++) {
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
        int min = dp[0];
        for (int i = 1; i < n; i++) {
            min = Math.min(min, dp[i]);
        }
        return min;
    }

    //[121].买卖股票的最佳时机
    public int maxProfit(int[] prices) {
        int n = prices.length;
        //到第i天，0表示不持股票，1表示持有股票
        int[][] dp = new int[n][2];

        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for (int i = 1; i < n; i++) {
            //不持股票，（昨天卖掉股票的利润，今天卖掉股票的利润）
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);

            //持有股票，（昨天持有股票的利润， 今天购入股票的利润）
            dp[i][1] = Math.max(dp[i - 1][1], -prices[i]);
        }

        return dp[n - 1][0];
    }

    //[121].买卖股票的最佳时机 优化版
    public int maxProfit_y(int[] prices) {
        int n = prices.length;
        int dp_0 = 0;
        int dp_1 = -prices[0];
        for (int i = 1; i < n; i++) {
            //不持股票，（今天卖掉股票的利润，昨天卖掉股票的利润）
            dp_0 = Math.max(dp_0, dp_1 + prices[i]);
            //持有股票，（昨天持有股票的利润， 今天购入股票的利润）
            dp_1 = Math.max(dp_1, -prices[i]);
        }

        return dp_0;
    }

    //[122].买卖股票的最佳时机 II
    public int maxProfit2(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for (int i = 1; i < n; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }
        return dp[n - 1][0];
    }

    //[122].买卖股票的最佳时机 II 优化版
    public int maxProfit2_y(int[] prices) {
        int n = prices.length;
        int dp_0 = 0;
        int dp_1 = -prices[0];
        for (int i = 1; i < n; i++) {
            int pre_dp_0 = dp_0;
            int pre_dp_1 = dp_1;
            dp_0 = Math.max(pre_dp_0, pre_dp_1 + prices[i]);
            dp_1 = Math.max(pre_dp_1, pre_dp_0 - prices[i]);
        }
        return dp_0;
    }

    //[123].买卖股票的最佳时机 III
    public int maxProfit3(int[] prices) {
        // dp[i][k][0] = Math.max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
        // dp[i][k][1] = Math.max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
        int n = prices.length;
        int[][][] dp = new int[n][3][2];
        dp[0][1][0] = 0;
        dp[0][2][0] = 0;
        dp[0][1][1] = -prices[0];
        dp[0][2][1] = -prices[0];

        for (int i = 1; i < n; i++) {
            dp[i][2][0] = Math.max(dp[i - 1][2][0], dp[i - 1][2][1] + prices[i]);
            dp[i][2][1] = Math.max(dp[i - 1][2][1], dp[i - 1][1][0] - prices[i]);

            dp[i][1][0] = Math.max(dp[i - 1][1][0], dp[i - 1][1][1] + prices[i]);
            dp[i][1][1] = Math.max(dp[i - 1][1][1], -prices[i]);
        }
        return dp[n - 1][2][0];
    }

    //[123].买卖股票的最佳时机 III 优化版
    public int maxProfit3_y(int[] prices) {
        int n = prices.length;
        int dp_1_0 = 0;
        int dp_1_1 = -prices[0];
        int dp_2_0 = 0;
        int dp_2_1 = -prices[0];
        for (int i = 1; i < n; i++) {
            dp_2_0 = Math.max(dp_2_0, dp_2_1 + prices[i]);
            dp_2_1 = Math.max(dp_2_1, dp_1_0 - prices[i]);
            dp_1_0 = Math.max(dp_1_0, dp_1_1 + prices[i]);
            dp_1_1 = Math.max(dp_1_1, -prices[i]);
        }
        return dp_2_0;
    }

    //[128].最长连续序列
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int res = 0;
        for (int num : set) {
            if (!set.contains(num - 1)) {
                int cur = num;
                int longest = 1;
                while (set.contains(cur + 1)) {
                    cur += 1;
                    longest += 1;
                }
                res = Math.max(res, longest);
            }
        }
        return res;
    }

    //[129].求根节点到叶节点数字之和
    public int sumNumbers(TreeNode root) {
        return dfsForSumNumbers(root, 0);
    }

    private int dfsForSumNumbers(TreeNode root, int preVal) {
        if (root == null) return 0;
        int cur = preVal * 10 + root.val;
        if (root.left == null && root.right == null) {
            return cur;
        }
        return dfsForSumNumbers(root.right, cur) + dfsForSumNumbers(root.left, cur);
    }

    //[130].被围绕的区域
    public void solve(char[][] board) {
        int m = board.length, n = board[0].length;
        //从边界出发找到O的替换掉
        for (int i = 0; i < m; i++) {
            dfsForSolve(board, i, 0);
            dfsForSolve(board, i, n - 1);
        }
        for (int j = 0; j < m; j++) {
            dfsForSolve(board, 0, j);
            dfsForSolve(board, m - 1, j);
        }
        //把边界的O修改成X
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                } else if (board[i][j] == 'Y') {
                    board[i][j] = 'O';
                }
            }
        }

    }

    private void dfsForSolve(char[][] board, int x, int y) {
        int m = board.length, n = board[0].length;
        if (x < 0 || y < 0 || x > m - 1 || y > n - 1) {
            return;
        }
        //不是水
        if (board[x][y] != 'O') {
            return;
        }
        //淹掉它
        board[x][y] = 'Y';
        dfsForSolve(board, x - 1, y);
        dfsForSolve(board, x + 1, y);
        dfsForSolve(board, x, y - 1);
        dfsForSolve(board, x, y + 1);
    }

    //[131].分割回文串
    public List<List<String>> partition(String s) {
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
        }
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                //三个字符，并且收尾相等，就是回文
                dp[i][j] = (dp[i + 1][j - 1] || j - i < 3) && s.charAt(i) == s.charAt(j);
            }
        }

        List<List<String>> res = new ArrayList<>();
        dfsForPartition(s, 0, res, new LinkedList<>(), dp);
        return res;
    }

    private void dfsForPartition(String s, int start, List<List<String>> res, LinkedList<String> path, boolean[][] dp) {
        if (start == s.length()) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = start; i < s.length(); i++) {
            if (!dp[start][i]) {
                continue;
            }
            path.addLast(s.substring(start, i + 1));
            dfsForPartition(s, i + 1, res, path, dp);
            path.removeLast();
        }
    }

    //[133].克隆图
    public Node cloneGraph(Node node) {
        if (node == null) return null;
        //hash赋值，同时充当访问标记
        Map<Node, Node> cloneMap = new HashMap<>();
        dfsForCloneGraph(node, cloneMap);
        return cloneMap.get(node);
    }

    private void dfsForCloneGraph(Node node, Map<Node, Node> cloneMap) {
        //已经创建过了，就不需要再次创建了
        if (cloneMap.containsKey(node)) {
            return;
        }
        Node clone = new Node(node.val);
        cloneMap.put(node, clone);
        if (node.neighbors != null && node.neighbors.size() > 0) {
            clone.neighbors = new ArrayList<>();
            for (Node neigh : node.neighbors) {
                dfsForCloneGraph(neigh, cloneMap);
                clone.neighbors.add(cloneMap.get(neigh));
            }
        }
    }

    //[134].加油站
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int left = 0, minLeft = Integer.MAX_VALUE;
        int n = gas.length;
        int start = 0;
        for (int i = 0; i < n; i++) {
            left += gas[i] - cost[i];
            //剩余最小，更新位置
            if (left < minLeft) {
                minLeft = left;
                start = i;
            }
        }
        //下一个位置才是起始位置，并且可能超过数组长度，并且是环
        return left >= 0 ? (start + 1) % n : -1;
    }

    //[136].只出现一次的数字
    public int singleNumber(int[] nums) {
        int res = nums[0];
        for (int i = 1; i < nums.length; i++) {
            res ^= nums[i];
        }
        return res;
    }

    //[137].只出现一次的数字 II
    public int singleNumber2(int[] nums) {
        int res = 0;
        for (int i = 0; i < 32; i++) {
            int count = 0;
            int pos = 1 << i;
            for (int num : nums) {
                if ((pos & num) == pos) {
                    count++;
                }
            }
            if (count % 3 != 0) {
                res |= 1 << i;
            }
        }
        return res;
    }

    //[138].复制带随机指针的链表
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        //旧 + 新的
        Map<Node, Node> map = new HashMap<>();
        Node newHead = new Node(head.val);
        map.put(head, newHead);

        Node cur = head;
        while (cur != null) {
            Node copy = map.get(cur);

            if (cur.random != null) {
                map.putIfAbsent(cur.random, new Node(cur.random.val));
                copy.random = map.get(cur.random);
            }

            if (cur.next != null) {
                map.putIfAbsent(cur.next, new Node(cur.next.val));
                copy.next = map.get(cur.next);
            }

            cur = cur.next;
        }
        return newHead;
    }

    //[139].单词拆分
    public boolean wordBreak(String s, List<String> wordDict) {
        int n = s.length();
        //前i个字母是否可以拆分
        boolean[] dp = new boolean[n + 1];
        //后续的状态依赖dp[0]这个状态位，所以得修改语义为个数，而不是结尾
        dp[0] = true;
        for (int i = 1; i <= n; i++) {
            for (int j = i - 1; j >= 0; j--) {
                dp[i] = dp[j] && wordDict.contains(s.substring(j, i));
                if (dp[i]) break;
            }
        }
        return dp[n];
    }

    //[141].环形链表
    public boolean hasCycle(ListNode head) {
        ListNode slow = head, fast = head;

        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) return true;
        }
        return false;
    }

    //[142].环形链表 II
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) {
                break;
            }
        }
        if (fast == null || fast.next == null) return null;

        slow = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }

    //[143].重排链表
    public void reorderList(ListNode head) {
        if (head == null) return;
        ListNode slow = head, fast = head, mid = null;
        while (fast != null && fast.next != null) {
            mid = slow;
            fast = fast.next.next;
            slow = slow.next;
        }

        //奇数节点需要重置下
        if (fast != null) {
            mid = slow;
        }

        ListNode q = reverseList(mid.next);
        mid.next = null;
        ListNode p = head;
        while (q != null) {
            ListNode qNext = q.next;
            ListNode pNext = p.next;
            q.next = pNext;
            p.next = q;
            q = qNext;
            p = pNext;
        }
    }

    //[144].二叉树的前序遍历
    public List<Integer> preorderTraversal(TreeNode root) {
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
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
//        TreeNode cur = root;
//        while (cur != null || !stack.isEmpty()) {
//            if (cur != null) {
//                res.add(0, cur.val);
//                stack.push(cur);
//                cur = cur.right;
//            } else {
//                cur = stack.pop();
//                cur = cur.left;
//            }
//        }

        stack.push(root);
        while (!stack.isEmpty()) {
            root = stack.pop();
            //根右左，利用栈的后进先出能力，到下左右即可
            res.add(0, root.val);
            if (root.left != null) {
                stack.push(root.left);
            }
            if (root.right != null) {
                stack.push(root.right);
            }
        }
        return res;
    }

    //[146].LRU 缓存
    public class LRUCache {
        LinkedHashMap<Integer, Integer> cache;
        int capacity;

        public LRUCache(int capacity) {
            this.capacity = capacity;
            this.cache = new LinkedHashMap<>();
        }

        public int get(int key) {
            if (!cache.containsKey(key)) {
                return -1;
            }
            int val = cache.get(key);
            cache.remove(key);
            cache.put(key, val);
            return val;
        }

        public void put(int key, int value) {
            if (cache.containsKey(key)) {
                int val = cache.get(key);
                cache.remove(key);
                cache.put(key, val);
            } else {
                if (cache.size() >= capacity) {
                    cache.remove(cache.keySet().iterator().next());
                    cache.put(key, value);
                } else {
                    cache.put(key, value);
                }
            }
        }
    }

    //[147].对链表进行插入排序
    public ListNode insertionSortList(ListNode head) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode cur = head.next, lastSorted = head;
        while (cur != null) {
            //维护最后排序好的指针
            if (lastSorted.val <= cur.val) {
                lastSorted = lastSorted.next;
            } else {
                //肯定要从头开始找
                ListNode pre = dummy;
                while (pre.next.val < cur.val) {
                    pre = pre.next;
                }
                //lastSorted后继节点指向cur后面的节点，因为cur之前都是排序好的
                lastSorted.next = cur.next;

                //pre的后面一个节点比较大，插入到pre后面
                cur.next = pre.next;
                pre.next = cur;
            }
            cur = lastSorted.next;
        }
        return dummy.next;
    }

    //[148].排序链表
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode fast = head, slow = head, pre = null;
        while (fast != null && fast.next != null) {
            pre = slow;
            fast = fast.next.next;
            slow = slow.next;
        }
        pre.next = null;

        //奇数情况，左边少一个， 右边多一个
        ListNode left = sortList(head);
        ListNode right = sortList(slow);
        return merge(left, right);
    }

    private ListNode merge(ListNode left, ListNode right) {
        ListNode dummy = new ListNode(-1), cur = dummy;
        while (left != null && right != null) {
            if (left.val < right.val) {
                cur.next = left;
                left = left.next;
            } else {
                cur.next = right;
                right = right.next;
            }
            cur = cur.next;
        }

        cur.next = left == null ? right : left;
        return dummy.next;
    }

    //[150].逆波兰表达式求值
    public int evalRPN(String[] tokens) {
        Stack<String> stack = new Stack<>();
        for (String token : tokens) {
            if (token.equals("+")
                    || token.equals("-")
                    || token.equals("*")
                    || token.equals("/")) {
                int second = Integer.parseInt(stack.pop());
                int first = Integer.parseInt(stack.pop());
                if (token.equals("+")) {
                    stack.push("" + (second + first));
                } else if (token.equals("-")) {
                    stack.push("" + (first - second));
                } else if (token.equals("*")) {
                    stack.push("" + (first * second));
                } else {
                    stack.push("" + (first / second));
                }
            } else {
                stack.push(token);
            }
        }
        return stack.isEmpty() ? 0 : Integer.parseInt(stack.pop());
    }

    //[151].翻转字符串里的单词
    public String reverseWords(String s) {
        StringBuilder sb = new StringBuilder();
        int len = s.length();
        int left = 0;
        while (s.charAt(left) == ' ') {
            left++;
        }

        for (int i = len - 1; i >= left; i--) {
            int j = i;
            while (i >= left && s.charAt(i) != ' ') {
                i--;
            }

            if (i != j) {
                sb.append(s.substring(i + 1, j + 1));
                if (i > left) {
                    sb.append(" ");
                }
            }
        }
        return sb.toString();
    }

    //[152].乘积最大子数组
    public int maxProduct(int[] nums) {
        int n = nums.length;
        //以[i][0]为结尾的最大乘积,以[i][1]为结尾的最小乘积
        int[][] dp = new int[n][2];
        dp[0][0] = nums[0];
        dp[0][1] = nums[0];
        int max = nums[0];
        for (int i = 1; i < n; i++) {
            int num = nums[i];
            //因为是“连续”子数组，所以i位置只取决于i-1位置
            if (num > 0) {
                dp[i][0] = Math.max(dp[i - 1][0] * num, num);
                dp[i][1] = Math.min(dp[i - 1][1] * num, num);
            } else {
                dp[i][0] = Math.max(dp[i - 1][1] * num, num);
                dp[i][1] = Math.min(dp[i - 1][0] * num, num);
            }
            max = Math.max(max, dp[i][0]);
        }
        return max;
    }

    //[153].寻找旋转排序数组中的最小值
    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return nums[left];
    }

    //[160].相交链表
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode p1 = headA, p2 = headB;
        while (p1 != p2) {
            if (p1 != null) {
                p1 = p1.next;
            } else {
                p1 = headB;
            }

            if (p2 != null) {
                p2 = p2.next;
            } else {
                p2 = headA;
            }
        }
        return p1;
    }

    //[162].寻找峰值
    public int findPeakElement(int[] nums) {
        int left = 0, right = nums.length - 1;
        //因为需要判断后一个值，所以此处是left < right
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    //[165].比较版本号
    public int compareVersion(String version1, String version2) {
        int i = 0, j = 0, m = version1.length(), n = version2.length();
        //双指针要求同步进行，并且也全部遍历完，所以用或
        while (i < m || j < n) {
            int x = 0;
            for (; i < m && version1.charAt(i) != '.'; i++) {
                x = x * 10 + version1.charAt(i) - '0';
            }
            //跳过.号
            i++;
            int y = 0;
            for (; j < n && version2.charAt(j) != '.'; j++) {
                y = y * 10 + version2.charAt(j) - '0';
            }
            j++;
            if (x > y) {
                return 1;
            } else if (x < y) {
                return -1;
            }
        }
        return 0;
    }

    //[166].分数到小数
    public String fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) return "0";
        long remain = Math.abs(numerator);
        long div = Math.abs(denominator);
        StringBuilder sb = new StringBuilder();
        if (numerator < 0 ^ denominator < 0) {
            sb.append('-');
        }
        //先做整数部分的除法操作
        sb.append(remain / div);
        remain %= div;
        if (remain == 0) return sb.toString();
        sb.append('.');

        //再做小数部分的除法操作
        Map<Long, Integer> index = new HashMap<>();
        while (remain != 0) {
            //涉及到分数的运算了，所以直接可以通过有没有来判断
            if (index.containsKey(remain)) {
                int idx = index.get(remain);
                sb.insert(idx, '(');
                sb.append(')');
                break;
            } else {
                //每次都是余数数字，记录需要插入的位置
                index.put(remain, sb.length());
                //每次都要扩10倍，模拟竖式除法
                remain *= 10;
                sb.append(remain / div);
                remain %= div;
            }
        }
        return sb.toString();
    }

    //[169].多数元素
    public int majorityElement(int[] nums) {
        int candidate = -1;
        int count = 0;
        for (int num : nums) {
            if (count != 0 && candidate == num) count++;
            else if (count == 0) {
                candidate = num;
                count++;
            } else {
                count--;
            }
        }
        return candidate;
    }

    //[172].阶乘后的零
    public int trailingZeroes(int n) {
        int res = 0;
        long d = 5;
        while (n >= d) {
            res += n / d;
            d *= 5;
        }
        return res;
    }

    //[173].二叉搜索树迭代器
    public class BSTIterator {
        Stack<TreeNode> stack;

        public BSTIterator(TreeNode root) {
            stack = new Stack<>();
            TreeNode cur = root;
            //中序遍历的迭代算法拆成了两半，前半段压栈，后半段弹栈
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
        }

        public int next() {
            TreeNode cur = stack.pop();
            cur = cur.right;
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            return cur.val;
        }

        public boolean hasNext() {
            return !stack.isEmpty();
        }
    }

    //[179].最大数
    public String largestNumber(int[] nums) {
        int n = nums.length;
        String[] ans = new String[n];
        for (int i = 0; i < n; i++) {
            ans[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(ans, (a, b) -> (b + a).compareTo(a + b));
        if (ans[0].equals("0")) return "0";
        return String.join("", ans);
    }

    //[187].重复的DNA序列
    public List<String> findRepeatedDnaSequences(String s) {
        List<String> res = new ArrayList<>();
        if (s.length() < 10) return res;
//        Map<String, Integer> dnaMap = new HashMap<>();
//        for (int i = 10; i < s.length(); i++) {
//            String subDna = s.substring(i - 10, i);
//            dnaMap.put(subDna, dnaMap.getOrDefault(subDna, 0) + 1);
//            //这是不是也是一个思路改进呢？？
//            if (dnaMap.get(subDna) == 2) {
//                res.add(subDna);
//            }
//        }

        Map<Character, Integer> map = new HashMap<>();
        map.put('A', 0);
        map.put('C', 1);
        map.put('T', 2);
        map.put('G', 3);
        //前10位先构造出来，一共需要20位，理论上够的
        int x = 0;
        for (int i = 0; i < 9; i++) {
            x = (x << 2) | map.get(s.charAt(i));
        }
        int n = s.length();
        Map<Integer, Integer> count = new HashMap<>();
        for (int i = 0; i <= n - 10; i++) {
            //前面算了9位，从index 9开始算
            x = ((x << 2) | map.get(s.charAt(i + 9))) & ((1 << 20) - 1);
            count.put(x, count.getOrDefault(x, 0) + 1);
            if (count.get(x) == 2) {
                res.add(s.substring(i, i + 10));
            }
        }
        return res;
    }

    //[198].打家劫舍
    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 0) return 0;
        if (n == 1) return nums[0];

        //dp[i] 从第i间房打劫的最大金额
        int[] dp = new int[n + 1];
        dp[n - 1] = nums[n - 1];
        //因为i需要从倒数第二个位置开始，所以需要n+1个空间
        for (int i = n - 2; i >= 0; i--) {
            dp[i] = Math.max(dp[i + 2] + nums[i], dp[i + 1]);
        }
        return dp[0];
    }

    //[199].二叉树的右视图
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        dfsForRightSideView(root, 0, res);
        return res;
    }

    private void dfsForRightSideView(TreeNode root, int depth, List<Integer> res) {
        if (root == null) return;
        if (res.size() == depth) {
            res.add(root.val);
        }
        dfsForRightSideView(root.right, depth + 1, res);
        dfsForRightSideView(root.left, depth + 1, res);
    }

    //[200].岛屿数量
    public int numIslands(char[][] grid) {
        int m = grid.length, n = grid[0].length;
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    res++;
                    dfsForIslands(grid, i, j);
                }
            }
        }
        return res;
    }

    private void dfsForIslands(char[][] grid, int x, int y) {
        int m = grid.length, n = grid[0].length;
        if (x < 0 || x > m - 1 || y < 0 || y > n - 1) {
            return;
        }

        if (grid[x][y] == '0') {
            return;
        }
        //是陆地就淹掉它
        grid[x][y] = '0';
        dfsForIslands(grid, x - 1, y);
        dfsForIslands(grid, x + 1, y);
        dfsForIslands(grid, x, y - 1);
        dfsForIslands(grid, x, y + 1);
    }

    //[201].数字范围按位与
    public int rangeBitwiseAnd(int left, int right) {
        int m = left, n = right;
        int count = 0;
        while (m < n) {
            count++;
            m >>= 1;
            n >>= 1;
        }
        return m << count;
    }


    //[203].移除链表元素
    public ListNode removeElements(ListNode head, int val) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode slow = dummy, fast = head;
        while (fast != null) {
            if (fast.val == val) {
                slow.next = fast.next;
            } else {
                slow = fast;
            }
            fast = fast.next;
        }
        return dummy.next;
    }

    //[206].反转链表
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode last = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return last;
    }

    //[206].反转链表 迭代
    private ListNode reverseList2(ListNode head) {
        ListNode cur = head, pre = null;
        while (cur != null) {
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }

    //[207].课程表
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        //入度，找到入度为0的节点，然后依次遍历，减度数，如果为入度为0加入
        int[] indegree = new int[numCourses];
        for (int[] pre : prerequisites) {
            indegree[pre[0]]++;
        }

        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < indegree.length; i++) {
            if (indegree[i] == 0) {
                queue.offer(i);
            }
        }
        int count = 0;
        while (!queue.isEmpty()) {
            int course = queue.poll();
            count++;
            for (int[] pre : prerequisites) {
                if (pre[1] != course) continue;
                indegree[pre[0]]--;

                //只有入度为0的点才加进去
                if (indegree[pre[0]] == 0) {
                    queue.offer(pre[0]);
                }
            }
        }
        return count == numCourses;
    }

    //[208].实现 Trie (前缀树)
    public static class Trie {

        private Trie[] children;
        private boolean isEnd;

        public Trie() {
            children = new Trie[26];
            isEnd = false;
        }

        public void insert(String word) {
            Trie cur = this;
            for (int i = 0; i < word.length(); i++) {
                int ch = word.charAt(i) - 'a';
                if (cur.children[ch] == null) {
                    cur.children[ch] = new Trie();
                }
                cur = cur.children[ch];
            }
            cur.isEnd = true;
        }

        public boolean search(String word) {
            if (word == null || word.length() == 0) return false;
            Trie cur = this;
            for (int i = 0; i < word.length(); i++) {
                int ch = word.charAt(i) - 'a';
                if (cur.children[ch] == null) {
                    return false;
                }
                cur = cur.children[ch];
            }
            return cur.isEnd;
        }

        public boolean startsWith(String prefix) {
            if (prefix == null || prefix.length() == 0) return false;
            Trie cur = this;
            for (int i = 0; i < prefix.length(); i++) {
                int ch = prefix.charAt(i) - 'a';
                if (cur.children[ch] == null) {
                    return false;
                }
                cur = cur.children[ch];
            }
            return true;
        }
    }

    //[209].长度最小的子数组
    public int minSubArrayLen(int target, int[] nums) {
        //2,3,1,2,4, 3
        //2 5 6 8 12 15
        int n = nums.length;
        int[] preSum = new int[n + 1];
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += nums[i];
            preSum[i + 1] = sum;
        }
        int left = 0, right = 0, res = Integer.MAX_VALUE;
        while (right < nums.length) {
            right++;
            //找到合理的，就缩
            while (preSum[right] - preSum[left] >= target) {
                res = Math.min(res, right - left);
                left++;
            }

        }
        return res == Integer.MAX_VALUE ? 0 : res;
    }

    //[210].课程表 II
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] indeg = new int[numCourses];
        for (int[] pre : prerequisites) {
            indeg[pre[0]]++;
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indeg[i] == 0) {
                queue.offer(i);
            }
        }

        int[] res = new int[numCourses];
        int index = 0;
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            res[index++] = cur;
            for (int[] pre : prerequisites) {
                if (pre[1] != cur) continue;
                if (--indeg[pre[0]] == 0) {
                    queue.offer(pre[0]);
                }
            }

        }
        return index == numCourses ? res : new int[0];
    }

    //[213].打家劫舍 II
    public int rob2(int[] nums) {
        int len = nums.length;
        if (len == 0) return 0;
        if (len == 1) return nums[0];
        return Math.max(rob2(nums, 0, len - 2), rob2(nums, 1, len - 1));
    }

    private int rob2(int[] nums, int start, int end) {
        int dp_i = 0, dp_i_2 = 0, dp_i_1 = 0;
        for (int i = end; i >= start; i--) {
            dp_i = Math.max(dp_i_2 + nums[i], dp_i_1);
            dp_i_2 = dp_i_1;
            dp_i_1 = dp_i;
        }
        return dp_i;
    }

    //[215].数组中的第K个最大元素
    public int findKthLargest(int[] nums, int k) {
        //第K大意味着是从小到大是第n-k位
        return quickSort(nums, 0, nums.length - 1, nums.length - k);
    }

    private int quickSort(int[] nums, int left, int right, int targetIndex) {
        int l = left, r = right;
        int pivot = nums[l];
        while (l < r) {
            while (l < r && pivot <= nums[r]) r--;
            nums[l] = nums[r];

            while (l < r && nums[l] <= pivot) l++;
            nums[r] = nums[l];
        }
        nums[l] = pivot;

        if (l == targetIndex) {
            return nums[l];
        } else if (l < targetIndex) {
            return quickSort(nums, l + 1, right, targetIndex);
        } else {
            return quickSort(nums, left, l - 1, targetIndex);
        }
    }

    //[216].组合总和 III
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> res = new ArrayList<>();
        if (n < 0) return res;
        backtraceForCombinationSum3(k, n, 1, res, new LinkedList<>());
        return res;
    }

    private void backtraceForCombinationSum3(int k, int n, int s, List<List<Integer>> res, LinkedList<Integer> select) {
        if (select.size() == k) {
            if (n == 0) {
                res.add(new ArrayList<>(select));
            }
            return;
        }

        for (int i = s; i <= 9; i++) {
            if (i > n) {
                continue;
            }
            select.addLast(i);
            backtraceForCombinationSum3(k, n - i, i + 1, res, select);
            select.removeLast();
        }
    }

    //[217].存在重复元素
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (set.contains(num)) {
                return true;
            }
            set.add(num);
        }
        return false;
    }

    //[219].存在重复元素 II
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Set<Integer> window = new HashSet<>();
        //这个窗口每次只会增1， 缩窗口也只是每次减1，那么不需要两个指针
        for (int j = 0; j < nums.length; j++) {
            int num = nums[j];
            if (window.contains(num)) {
                return true;
            }
            window.add(num);
            //绝对差值为k，意味着窗口有k+1个元素，刚达到窗口边界，就需要缩左边界
            if (window.size() >= k + 1) {
                window.remove(nums[j - k]);
            }
        }
        return false;
    }

    //[220].存在重复元素 III
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
//        int n = nums.length;
//        TreeSet<Long> set = new TreeSet<>();
//        for (int i = 0; i < n; i++) {
//            //这边先操作，当i == k的时候
//            Long u = nums[i] * 1L;
//            Long l = set.floor(u);
//            Long r = set.ceiling(u);
//            if (l != null && u - l <= t) return true;
//            if (r != null && r - u <= t) return true;
//            set.add(u);
//            //下一次操作的时候，需要保证窗口内少一个元素，临界条件就是刚满足窗口就需要缩左边界
//            if (i >= k) set.remove(nums[i - k] * 1L);
//        }
//        return false;

        //桶排序思想来做
        //[0,1,2,3] t = 3 => 用t+1来划分桶号
        //[-4,-3,-2,-1] t = 3 => 用 ((nums[i] + 1) / (t + 1)) -1
        long size = t + 1L;
        int n = nums.length;
        Map<Long, Long> map = new HashMap<>();
        for (int i = 0; i < n; i++) {
            long num = nums[i] * 1L;
            long idx = getIdx(num, size);
            if (map.containsKey(idx)) return true;
            long l = idx - 1, r = idx + 1;
            if (map.containsKey(l) && num - map.get(l) <= t) return true;
            if (map.containsKey(r) && map.get(r) - num <= t) return true;

            map.put(idx, num);

            if (i >= k) map.remove(getIdx(nums[i - k], size));
        }
        return false;
    }

    private long getIdx(long num, long size) {
        if (num >= 0) return num / (size + 1);
        //+1使得负数往右边偏移一格，然后除以size，-3 ~ 0范围，又因为0被处理过，所以-1往左边偏移
        return ((num + 1) / size - 1);
    }

    //[221].最大正方形
    public int maximalSquare(char[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        //以i,j为终点的能构成正方形的最大边长
        int[][] dp = new int[m][n];
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
                    }
                    res = Math.max(dp[i][j], res);
                }
            }
        }
        return res * res;
    }

    //[222].完全二叉树的节点个数
    public int countNodes(TreeNode root) {
        if (root == null) return 0;
        int leftHeight = countLevel(root.left);
        int rightHeight = countLevel(root.right);
        //右边一定是满二叉树
        if (leftHeight > rightHeight) return 1 << rightHeight + countNodes(root.left);
            //左边一定是满二叉树
        else return 1 << leftHeight + countNodes(root.right);
    }

    private int countLevel(TreeNode root) {
        if (root == null) return 0;
        int level = 0;
        TreeNode left = root;
        //因为是完全二叉树，所以只需要计算左边的
        while (left != null) {
            level++;
            left = left.left;
        }
        return level;
    }

    //[223].矩形面积
    public int computeArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) {
        int total = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1);
        if (ax2 < bx1 || ax1 > bx2 || ay1 > by2 || by1 > ay2) {
            return total;
        }
        return total - (Math.min(ay2, by2) - Math.max(ay1, by1)) * (Math.min(ax2, bx2) - Math.max(ax1, bx1));
    }

    //[225].用队列实现栈
    public class MyStack {
        Queue<Integer> queue;

        public MyStack() {
            queue = new LinkedList<>();
        }

        public void push(int x) {
            int size = queue.size();
            queue.offer(x);
            for (int i = 0; i < size; i++) {
                queue.offer(queue.poll());
            }
        }

        public int pop() {
            return queue.poll();
        }

        public int top() {
            return queue.peek();
        }

        public boolean empty() {
            return queue.isEmpty();
        }
    }

    //[226].翻转二叉树
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;

        invertTree(root.left);
        invertTree(root.right);
        return root;
    }

    //[228].汇总区间
    public List<String> summaryRanges(int[] nums) {
        List<String> ans = new ArrayList<>();
        int i = 0;
        for (int j = 0; j < nums.length; j++) {
            //提前感知到和下一个节点的关系
            if (j + 1 == nums.length || nums[j] + 1 != nums[j + 1]) {
                if (i != j) {
                    ans.add(nums[i] + "->" + nums[j]);
                } else {
                    ans.add(nums[i] + "");
                }
                //下一个不相等的地方
                i = j + 1;
            }
        }
        return ans;
    }

    //[229].求众数II
    public List<Integer> majorityElement2(int[] nums) {
        //摩尔投票法
        int c1 = 0, c2 = 0;
        int cnt1 = 0, cnt2 = 0;
        for (int num : nums) {
            if (cnt1 != 0 && c1 == num) cnt1++;
            else if (cnt2 != 0 && c2 == num) cnt2++;
            else if (cnt1 == 0) {
                //选择第一个候选人 或者出现新的候选人
                cnt1 = 1;
                c1 = num;
            } else if (cnt2 == 0) {
                cnt2 = 1;
                c2 = num;
            } else {
                cnt1--;
                cnt2--;
            }
        }

        int n = nums.length;
        //重新统计阶段
        cnt1 = cnt2 = 0;
        for (int num : nums) {
            if (c1 == num) cnt1++;
            else if (c2 == num) cnt2++;
        }
        List<Integer> res = new ArrayList<>();
        if (cnt1 > n / 3) res.add(c1);
        if (cnt2 > n / 3) res.add(c2);
        return res;
    }

    //[230].二叉搜索树中第K小的元素
    public int kthSmallest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<>();
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (--k == 0) {
                break;
            }
            root = root.right;
        }
        return root.val;
    }

    //[232].用栈实现队列
    public class MyQueue {

        Stack<Integer> out;
        Stack<Integer> s2;

        public MyQueue() {
            out = new Stack<>();
            s2 = new Stack<>();
        }

        public void push(int x) {
            //每次out栈保持栈顶是第一个元素，借助s2，转储下
            while (!out.isEmpty()) {
                s2.push(out.pop());
            }
            out.push(x);
            while (!s2.isEmpty()) {
                out.push(s2.pop());
            }
        }

        public int pop() {
            return out.pop();
        }

        public int peek() {
            return out.peek();
        }

        public boolean empty() {
            return out.isEmpty();
        }
    }

    //[234].回文链表
    private ListNode left;

    public boolean isPalindrome(ListNode head) {
        left = head;
        return helperIsPalindrome(head);
    }

    private boolean helperIsPalindrome(ListNode right) {
        if (right == null) return true;
        boolean res = helperIsPalindrome(right.next);
        res = res && (left.val == right.val);
        left = left.next;
        return res;
    }

    //[235].二叉搜索树的最近公共祖先
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root.val > q.val && root.val > p.val) return lowestCommonAncestor(root.left, p, q);
        else if (root.val < p.val && root.val < q.val)
            return lowestCommonAncestor(root.right, p, q);
        else return root;
    }

    //[236].二叉树的最近公共祖先
    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return root;
        //只要有一个节点跟父节点相等，就认为找到了一个潜在的祖先节点
        if (root == p) return root;
        if (root == q) return root;
        TreeNode leftCommon = lowestCommonAncestor2(root.left, p, q);
        TreeNode rightCommon = lowestCommonAncestor2(root.right, p, q);
        if (leftCommon == null) return rightCommon;
        if (rightCommon == null) return leftCommon;
        //左右两个祖先的时候，父节点才是公共祖先
        return root;
    }

    //[237].删除链表中的节点
    public void deleteNode(ListNode node) {
        //不知道前面的节点，那只能去复制，删掉下一个节点
        node.val = node.next.val;
        node.next = node.next.next;
    }

    //[238].除自身以外数组的乘积
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] left = new int[n + 1];
        int[] right = new int[n + 1];
        left[0] = 1;
        right[n] = 1;
        for (int i = 0; i < n; i++) {
            left[i + 1] = left[i] * nums[i];
        }
        for (int i = n - 1; i >= 0; i--) {
            right[i] = right[i + 1] * nums[i];
        }

        int[] res = new int[n];
        for (int i = 0; i < n; i++) {
            res[i] = left[i] * right[i + 1];
        }
        return res;
    }

    //[239].滑动窗口最大值
    public static int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        int[] ans = new int[n - k + 1];
        //单调队列
        LinkedList<Integer> queue = new LinkedList<>();
        for (int r = 0; r < n; r++) {
            while (!queue.isEmpty() && nums[queue.peekLast()] < nums[r]) {
                queue.pollLast();
            }
            queue.offerLast(r);

            //超出范围的就移除左边的值
            if (queue.peekFirst() <= r - k) {
                queue.pollFirst();
            }
            //窗口长度大于k的时候更新值
            if (r + 1 >= k) {
                ans[r + 1 - k] = nums[queue.peekFirst()];
            }
        }
        return ans;
    }

    //[240].搜索二维矩阵 II
    public boolean searchMatrix2(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int x = 0, y = n - 1;

        while (x >= 0 && x < m && y >= 0 && y < n) {
            if (matrix[x][y] == target) {
                return true;
            } else if (matrix[x][y] < target) {
                x++;
            } else {
                y--;
            }
        }
        return false;
    }

    //[241].为运算表达式设计优先级
    public List<Integer> diffWaysToCompute(String expression) {
        return dfsForDiffWaysToCompute(expression, new HashMap<>());
    }

    private List<Integer> dfsForDiffWaysToCompute(String expression, Map<String, List<Integer>> temp) {
        if (temp.containsKey(expression)) return temp.get(expression);
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < expression.length(); i++) {
            if (Character.isDigit(expression.charAt(i))) {
                continue;
            }
            List<Integer> first = dfsForDiffWaysToCompute(expression.substring(0, i), temp);
            List<Integer> second = dfsForDiffWaysToCompute(expression.substring(i + 1), temp);
            for (int l : first) {
                for (int r : second) {
                    if (expression.charAt(i) == '+') {
                        res.add(l + r);
                    } else if (expression.charAt(i) == '-') {
                        res.add(l - r);
                    } else {
                        res.add(l * r);
                    }
                }
            }
        }

        if (res.size() == 0) {
            res.add(Integer.parseInt(expression));
        }
        temp.put(expression, res);
        return res;
    }

    //[257].二叉树的所有路径
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> res = new ArrayList<>();
        //这道题目其实讲解了回溯的本质，每次递归之后，变量都需要回滚一次。回到递归之前的那个状态
//        dfsForBinaryTreePaths(root, new LinkedList<>(), res);
        //而用一个不可变量，形参不会发生改变，所以不需要回滚状态
        dfsForBinaryTreePaths(root, "", res);
        return res;
    }

    private void dfsForBinaryTreePaths(TreeNode root, LinkedList<String> select, List<String> res) {
        if (root == null) {
            return;
        }
        select.addLast(root.val + "");
        //叶子结点肯定退出
        if (root.left == null && root.right == null) {
            res.add(String.join("->", select));
            return;
        }
        //所以removeLast可以多次执行，只要递归多少次，就得删多少次
        //左子树不为空遍历
        if (root.left != null) {
            dfsForBinaryTreePaths(root.left, select, res);
            //删除的是退出递归的前一个元素
            select.removeLast();
        }
        //右子树不为空遍历
        if (root.right != null) {
            dfsForBinaryTreePaths(root.right, select, res);
            //删除的是退出递归的前一个元素
            select.removeLast();
        }
    }

    private void dfsForBinaryTreePaths(TreeNode root, String path, List<String> res) {
        if (root == null) return;

        path += root.val;
        if (root.left == null && root.right == null) {
            res.add(path);
            return;
        }

        dfsForBinaryTreePaths(root.left, path + "->", res);
        dfsForBinaryTreePaths(root.right, path + "->", res);
    }

    //[260].只出现一次的数字 III
    public int[] singleNumber3(int[] nums) {
        int xor = 0;
        for (int num : nums) xor ^= num;

        int k = 1;
        while ((xor & 1) == 0) {
            xor >>= 1;
            k <<= 1;
        }
        int res1 = 0, res2 = 0;
        for (int num : nums) {
            if ((num & k) == k) {
                res1 ^= num;
            } else {
                res2 ^= num;
            }
        }
        return new int[]{res1, res2};
    }

    //[261].以图判树
    public boolean validTree(int n, int[][] edges) {
        UnionFind uf = new UnionFind(n);
        for (int[] edge : edges) {
            if (uf.connect(edge[0], edge[1])) {
                return false;
            }
            uf.union(edge[0], edge[1]);
        }
        return uf.count == 1;
    }

    //[263].丑数
    public boolean isUgly(int num) {
        if (num < 0) return false;
        int remain = num;
        while (remain != 1) {
            if (remain % 2 == 0) {
                remain /= 2;
            } else if (remain % 3 == 0) {
                remain /= 3;
            } else if (remain % 5 == 0) {
                remain /= 5;
            } else {
                return false;
            }
        }
        return true;
    }

    //[264].丑数 II
    public int nthUglyNumber(int n) {
        //2 3 5 就是不用动态规划
        int[] factors = new int[]{2, 3, 5};
        PriorityQueue<Long> queue = new PriorityQueue<>();
        Set<Long> set = new HashSet<>();
        queue.offer(1L);
        set.add(1L);
        int res = 0;
        for (int i = 0; i < n; i++) {
            long min = queue.poll();
            res = (int) min;
            for (int factor : factors) {
                long value = min * factor;
                if (!set.contains(value)) {
                    set.add(value);
                    queue.offer(value);
                }
            }
        }
        return res;
    }

    //[274].H 指数
    public int hIndex1(int[] citations) {
        //0 1 3 5 6
        Arrays.sort(citations);
        int count = 0, n = citations.length;
        while (count < n && citations[n - count - 1] > count) {
            count++;
        }
        return count;
    }

    //[275].H 指数 II
    public int hIndex(int[] citations) {
        int n = citations.length;
        int left = 0, right = n - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            //引用次数小于论文篇数(不满足共h篇论文引用次数不少于h)，往右边走
            if (citations[mid] < n - mid) {
                left = mid + 1;
            } else {
                //引用次数>=论文篇数，还应该往左边走走，找到更大的论文篇数
                right = mid;
            }
        }
        //h指数
        return n - left;
    }

    //[279].完全平方数
    public int numSquares(int n) {
        if (n <= 0) return 0;
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            dp[i] = Integer.MAX_VALUE;
            for (int j = 1; j * j <= i; j++) {
                dp[i] = Math.min(dp[i - j * j] + 1, dp[i]);
            }
        }
        return dp[n];
    }

    //[283].移动零
    public void moveZeroes(int[] nums) {
        int n = nums.length;
        int left = 0;
        for (int right = 0; right < n; right++) {
            if (nums[right] != 0) {
                nums[left] = nums[right];
                left++;
            }
        }
        //减少交换次数
        while (left < n) {
            nums[left++] = 0;
        }
    }

    //[287].寻找重复数
    public int findDuplicate(int[] nums) {
        //这个题目与442区分开，287只有一个数是重复数，题目不允许修改数组；442是有多个数，可以修改数组来标记是否重复
        //把slow当成是链表看待
        int slow = 0, fast = 0;
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);

        slow = 0;
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }

    //[297].二叉树的序列化与反序列化
    public class Codec {

        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            if (root == null) return "#";
            //先序遍历
            return root.val + "," + serialize(root.left) + "," + serialize(root.right);
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            LinkedList<String> list = new LinkedList<>();
            String[] items = data.split(",");
            for (String item : items) {
                list.addLast(item);
            }
            return dfsForDeserialize(list);
        }

        private TreeNode dfsForDeserialize(LinkedList<String> list) {
            if (list.isEmpty()) return null;
            String first = list.removeFirst();
            if (first.equals("#")) {
                return null;
            }
            TreeNode root = new TreeNode(Integer.parseInt(first));
            root.left = dfsForDeserialize(list);
            root.right = dfsForDeserialize(list);
            return root;
        }
    }

    //[299].猜数字游戏
    public String getHint(String secret, String guess) {
        int[] cnt1 = new int[10], cnt2 = new int[10];
        int A = 0, B = 0, n = guess.length();
        for (int i = 0; i < n; i++) {
            int c1 = secret.charAt(i) - '0', c2 = guess.charAt(i) - '0';
            if (c1 == c2) {
                A++;
            } else {
                cnt1[c1]++;
                cnt2[c2]++;
            }
        }
        //取最小的数字
        for (int i = 0; i <= 9; i++) B += Math.min(cnt1[i], cnt2[i]);
        return A + "A" + B + "B";
    }

    //[300].最长递增子序列
    public int lengthOfLIS(int[] nums) {
        int size = nums.length;
        if (size == 0) return 0;
        int[] dp = new int[size];
        int length = 1;
        for (int i = 0; i < size; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            length = Math.max(length, dp[i]);
        }
        return length;
    }

    private int lengthOfLIS2(int[] nums) {
        int size = nums.length;
        if (size == 0) return 0;
        //最大递增子序列
        int[] dp = new int[size];
        //最大默认有0堆
        int maxLen = 0;
        for (int num : nums) {
            //右侧边界，大于的模版
            int left = 0, right = maxLen;
            while (left < right) {
                int mid = left + (right - left) / 2;
                if (dp[mid] <= num) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            //放最顶上去
            dp[left] = num;
            //需要新建一个堆
            if (maxLen == left) {
                maxLen++;
            }
        }
        return maxLen;
    }

    //[303].区域和检索 - 数组不可变
    public static class NumArray {

        //preSum[i]代表nums[0...i-1]的和
        int[] preSum;

        public NumArray(int[] nums) {
            preSum = new int[nums.length + 1];
            int sum = 0;
            for (int i = 0; i < nums.length; i++) {
                sum += nums[i];
                preSum[i + 1] = sum;
            }
        }

        public int sumRange(int left, int right) {
            return preSum[right + 1] - preSum[left];
        }
    }

    //[304].二维区域和检索 - 矩阵不可变
    public static class NumMatrix {
        int[][] preSum;

        public NumMatrix(int[][] matrix) {
            int m = matrix.length;
            int n = matrix[0].length;
            preSum = new int[m + 1][n + 1];

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    preSum[i + 1][j + 1] = preSum[i][j + 1] + preSum[i + 1][j] - preSum[i][j] + matrix[i][j];
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            //减数据的时候，要结合图理解下，应该是外面的一格坐标才行
            return preSum[row2 + 1][col2 + 1] - preSum[row2 + 1][col1] - preSum[row1][col2 + 1] + preSum[row1][col1];
        }
    }

    //[306].累加数
    public boolean isAdditiveNumber(String num) {
        //199100199
        //选择，前导0不能选择，需要知道之前两个数是什么，只能从前往后选择
        return dfsForIsAdditiveNumber(num, 0, 0, 0, 0);
    }

    private boolean dfsForIsAdditiveNumber(String num, int start, long first, long second, int k) {
        if (start == num.length()) {
            return k > 2;
        }

        for (int i = start; i < num.length(); i++) {
            long cur = selectNumber(num, start, i);
            if (cur == -1) {
                continue;
            }
            if (k >= 2 && first + second != cur) {
                continue;
            }

            if (dfsForIsAdditiveNumber(num, i + 1, second, cur, k + 1)) {
                return true;
            }
        }
        return false;
    }

    private long selectNumber(String num, int start, int end) {
        //单个0是一个累加序列合法值
        if (start < end && num.charAt(start) == '0') {
            return -1;
        }
        long number = 0;
        for (int i = start; i <= end; i++) {
            number = number * 10 + (num.charAt(i) - '0');
        }
        return number;
    }

    //[310].最小高度树
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if (n == 1) {
            return Arrays.asList(0);
        }
        int[] deg = new int[n];
        Map<Integer, List<Integer>> link = new HashMap<>();
        for (int[] edge : edges) {
            deg[edge[0]]++;
            deg[edge[1]]++;
            link.putIfAbsent(edge[0], new ArrayList<>());
            link.get(edge[0]).add(edge[1]);

            link.putIfAbsent(edge[1], new ArrayList<>());
            link.get(edge[1]).add(edge[0]);
        }
        Queue<Integer> queue = new LinkedList<>();
        //无向图的出度判断为1，有向图出度为0
        //叶子节点的出度为1
        for (int i = 0; i < deg.length; i++) {
            if (deg[i] == 1) {
                queue.offer(i);
            }
        }
        List<Integer> res = new ArrayList<>();
        while (!queue.isEmpty()) {
            int size = queue.size();
            res = new ArrayList<>();
            //最外面的一层，叶子节点全部移除.
            for (int i = 0; i < size; i++) {
                int cur = queue.poll();
                List<Integer> neighbours = link.get(cur);
                for (int neighbour : neighbours) {
                    if (--deg[neighbour] == 1) {
                        queue.offer(neighbour);
                    }
                }
                res.add(cur);
            }
        }
        return res;
    }

    //[313].超级丑数
    public int nthSuperUglyNumber(int n, int[] primes) {
        int k = primes.length;
        //质数的指针
        int[] pointers = new int[k];
        int[] dp = new int[n];
        dp[0] = 1;
        for (int i = 1; i < n; i++) {
            int min = Integer.MAX_VALUE;
            for (int j = 0; j < k; j++) {
                min = Math.min(dp[pointers[j]] * primes[j], min);
            }
            dp[i] = min;

            for (int j = 0; j < k; j++) {
                if (dp[pointers[j]] * primes[j] == min) {
                    pointers[j]++;
                }
            }
        }
        return dp[n - 1];
    }

    //[316].去除重复字母
    public String removeDuplicateLetters(String s) {
        int[] cnts = new int[26];
        for (char ch : s.toCharArray()) {
            cnts[ch - 'a']++;
        }

        Stack<Character> stack = new Stack<>();
        boolean[] instack = new boolean[26];
        for (char ch : s.toCharArray()) {
            cnts[ch - 'a']--;
            //重复的情况，在这里就被处理掉了
            if (instack[ch - 'a']) {
                continue;
            }
            while (!stack.isEmpty() && stack.peek() > ch) {
                if (cnts[stack.peek() - 'a'] == 0) {
                    break;
                }
                instack[stack.pop() - 'a'] = false;
                ;
            }

            stack.push(ch);
            instack[ch - 'a'] = true;
        }
        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            sb.insert(0, stack.pop());
        }
        return sb.toString();
    }

    //[318].最大单词长度乘积
    public int maxProduct(String[] words) {
        int n = words.length;
        if (n == 0) return 0;
        int[] hashs = new int[n];
        for (int i = 0; i < n; i++) {
            int hash = 0;
            for (char ch : words[i].toCharArray()) {
                hash |= 1 << (ch - 'a');
            }
            hashs[i] = hash;
        }
        int ans = 0;
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if ((hashs[i] & hashs[j]) == 0) {
                    ans = Math.max(ans, words[i].length() * words[j].length());
                }
            }
        }
        return ans;
    }

    //[319].灯泡开关
    public int bulbSwitch(int n) {
        //对于位置k而言，只要是k的约数那么就会对灯泡进行操作，如果是偶数个约数，位置k的灯泡就会关闭，否则开启。
        //例如对于1，4而言，都是有奇数个约数，其中4的约数有1，2，4，正好对位置4进行了3次操作。4其实就是一个完全平方数
        //1,4，9，16是完全平方数，对于n而言就是算有多少个完全平方数，即√n
        return (int) Math.sqrt(n);
    }

    //[326].3的幂
    public boolean isPowerOfThree(int n) {
        if (n <= 0) return false;
        // 45 = 3 * 3 * 5
        // 9 = 3 * 3 * 1
/*        while (n % 3 == 0) {
            n = n / 3;
        }
        return n == 1;*/

        //因为3是质数，所以3^19肯定是3^n的最大公约数。
        int max = (int) Math.pow(3, 19);
        return max % n == 0;
    }

    //[328].奇偶链表
    public ListNode oddEvenList(ListNode head) {
        ListNode oddHead = new ListNode(-1);
        ListNode evenHead = new ListNode(-1);
        ListNode cur = head, odd = oddHead, even = evenHead;
        for (int i = 1; cur != null; i++) {
            ListNode next = cur.next;
            if (i % 2 != 0) {
                odd.next = cur;
                odd = odd.next;
            } else {
                even.next = cur;
                even = even.next;
            }
            cur.next = null;
            cur = next;
        }
        odd.next = evenHead.next;
        return oddHead.next;
    }

    //[331].验证二叉树的前序序列化
    public boolean isValidSerialization(String preorder) {
//        //用a##替换成一个#, 最后只会剩下一个#
//        Stack<String> stack = new Stack<>();
//        String[] vals = preorder.split(",");
//        for (String val :vals) {
//            if (val.equals("#")) {
//                if (!stack.isEmpty() && stack.peek().equals("#")) {
//                    //遇到两个#的时候，栈中有一个，pop出来之后，如果栈为空 或者 栈没有数字就是非法
//                    stack.pop();
//                    if (stack.isEmpty() || stack.pop().equals("#")) return false;
//                }
//                //两个#变成一个#，或者一个#直接进栈
//                stack.push(val);
//            } else {
//                stack.push(val);
//            }
//        }
//        //正常情况栈中会有一个#，否则就是非法二叉树
//        return stack.size() == 1 && stack.peek().equals("#");

        //头结点是2出度，0入度， null节点是0出度，1入度，其他节点2出度，1入度
        String[] ss = preorder.split(",");
        int n = ss.length;
        int out = 0, in = 0;
        for (int i = 0; i < n; i++) {
            if (!ss[i].equals("#")) out += 2;
            if (i != 0) in += 1;
            //当入度>=出度的时候肯定不是正常二叉树
            if (i != n - 1 && in >= out) return false;
        }
        return in == out;
    }

    //[334].递增的三元子序列
    public boolean increasingTriplet(int[] nums) {
        int firstMin = Integer.MAX_VALUE, secondMin = Integer.MAX_VALUE;
        for (int num : nums) {
            if (num <= firstMin) {
                firstMin = num;
            } else if (num <= secondMin) {
                secondMin = num;
            } else {
                return true;
            }
        }
        return false;
    }

    //[337].打家劫舍 III
    private Map<TreeNode, Integer> map = new HashMap<>();

    public int rob(TreeNode root) {
        if (root == null) return 0;
        if (map.containsKey(root)) return map.get(root);
        int rob_it = root.val +
                (root.left != null ? rob(root.left.left) + rob(root.left.right) : 0) +
                (root.right != null ? rob(root.right.left) + rob(root.right.right) : 0);
        int rob_not = rob(root.left) + rob(root.right);
        int res = Math.max(rob_it, rob_not);
        map.put(root, res);
        return res;
    }

    //[338].比特位计数
    public int[] countBits(int n) {
        ////0 --> 0
        ////1 --> 1
        ////2 --> 10
        ////3 --> 11
        ////4 --> 100
        ////5 --> 101
        //偶数一定等于dp[x >> 1], 奇数等于dp[x >> 1] +1
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            //奇数
            dp[i] = dp[i >> 1] + ((i & 1) == 1 ? 1 : 0);
        }
        return dp;
    }

    //[342].4的幂
    public boolean isPowerOfFour(int n) {
        // 16 = 4*4*1
        if (n <= 0) return false;
        //2的幂， 而且n & 1010 1010 1010 1010 为0，偶数位为1
        return (n & (n - 1)) == 0 && (n & 0xaaaaaaaa) == 0;
    }

    //[343].整数拆分
    public int integerBreak(int n) {
        //1 2 3 4 5
        //0 1 2
        int[] dp = new int[n + 1];
        dp[1] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j < i; j++) {
                //之前的拆法的乘积*拆出来的数 直接拆出两个数的乘积
                dp[i] = Math.max(dp[i], Math.max((i - j) * dp[j], (i - j) * j));
            }
        }
        return dp[n];
    }

    //[344].反转字符串
    public void reverseString(char[] s) {
        int left = 0, right = s.length - 1;
        while (left < right) {
            char ch = s[left];
            s[left] = s[right];
            s[right] = ch;
            left++;
            right--;
        }
    }

    //[354].俄罗斯套娃信封问题
    public int maxEnvelopes(int[][] envelopes) {
        int n = envelopes.length;
        Arrays.sort(envelopes, (a, b) -> {
            if (a[0] == b[0]) return b[1] - a[1];
            else return a[0] - b[0];
        });
        int[] height = new int[n];
        for (int i = 0; i < n; i++) {
            height[i] = envelopes[i][1];
        }
        return lengthOfLIS(height);
    }

    //[371].两整数之和
    public int getSum(int a, int b) {
        //正负数相加没关系，运算是通过补码实现的。
        //正整数的补码等于原码，负整数的补码 = 符号位不变，其他位反码 + 1
        //异或为非进位相加，与为进位
        //指导进位为0，循环结束
        while (b != 0) {
            //非进位相加
            int xor = a ^ b;
            //进位信息
            int carry = (a & b) << 1;
            a = xor;
            b = carry;
        }
        return a;
    }

    //[372].超级次方
    public int superPow(int a, int[] b) {
        return dfsForSuperPow(a, b, b.length - 1);
    }

    private int dfsForSuperPow(int a, int[] b, int index) {
        if (index < 0) {
            return 1;
        }
        int base = 1337;
        int part1 = myPow(a, b[index]);
        int part2 = myPow(dfsForSuperPow(a, b, index - 1), 10);
        return part1 * part2 % base;
    }

    //快速幂
    private int myPow(int a, int b) {
        if (b == 0) return 1;
        int base = 1337;
        a %= base;

        if (b % 2 == 0) {
            int half = myPow(a, b / 2);
            return half * half % base;
        } else {
            return a * myPow(a, b - 1) % base;
        }
    }

    //[373].查找和最小的 K 对数字
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums1.length == 0 || nums2.length == 0) return res;
        //和作为排序依据
        PriorityQueue<int[]> queue = new PriorityQueue<>(Comparator.comparingInt(o -> nums1[o[0]] + nums2[o[1]]));
        for (int i = 0; i < Math.min(nums1.length, k); i++) {
            queue.offer(new int[]{i, 0});
        }
        while (k > 0 && !queue.isEmpty()) {
            int[] top = queue.poll();
            res.add(Arrays.asList(nums1[top[0]], nums2[top[1]]));
            if (top[1] + 1 < nums2.length) {
                queue.offer(new int[]{top[0], top[1] + 1});
            }
            k--;
        }
        return res;
    }

    //[374].猜数字大小
    public class Solution374 { //extends GuessGame {
        private int guess(int num) {
            return 1;
        }

        public int guessNumber(int n) {
            int l = 1, r = n;
            while (l <= r) {
                int mid = l + (r - l) / 2;
                int res = guess(mid);
                if (res == 0) {
                    return mid;
                } else if (res > 0) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
            return -1;
        }
    }

    //[380].O(1) 时间插入、删除和获取随机元素
    public class RandomizedSet {
        //只要是获取随机需要提前知道大小并且能根据index直接获取
        Map<Integer, Integer> map;
        List<Integer> list;
        Random random = new Random();

        public RandomizedSet() {
            map = new HashMap<>();
            list = new ArrayList<>();
        }

        public boolean insert(int val) {
            if (map.containsKey(val)) return false;

            map.put(val, list.size());
            list.add(list.size(), val);
            return true;
        }

        public boolean remove(int val) {
            if (!map.containsKey(val)) return false;

            int lastIdx = list.size() - 1;
            int lastVal = list.get(lastIdx);
            int idx = map.get(val);

            //交换最后一个值到idx上
            list.set(idx, lastVal);
            //修改最新元素索引
            map.put(lastVal, idx);

            //删除最后一个元素
            list.remove(lastIdx);
            //删除该元素
            map.remove(val);
            return true;
        }

        public int getRandom() {
            int idx = random.nextInt(list.size());
            return list.get(idx);
        }
    }

    //[382].链表随机节点
    public class Solution382 {
        ListNode head;

        public Solution382(ListNode head) {
            this.head = head;
        }

        public int getRandom() {
            ListNode cur = head;
            int idx = 0, ans = 0;
            Random random = new Random();
            while (cur != null) {
                //蓄水池抽样：未知的规模中等概率选择样本， 从[0,i)选择一个元素
                if (random.nextInt(++idx) == 0) {
                    ans = cur.val;
                }
                cur = cur.next;
            }
            return ans;
        }
    }

    //[383].赎金信
    public boolean canConstruct(String ransomNote, String magazine) {
        if (ransomNote.length() > magazine.length()) return false;
        int[] count = new int[26];
        for (int i = 0; i < magazine.length(); i++) {
            count[magazine.charAt(i) - 'a']++;
        }
        for (int i = 0; i < ransomNote.length(); i++) {
            if (--count[ransomNote.charAt(i) - 'a'] < 0) {
                return false;
            }
        }
        return true;
    }

    //[384].打乱数组
    class Solution384 {
        int[] original;

        public Solution384(int[] nums) {
            original = nums;
        }

        public int[] reset() {
            return original;
        }

        public int[] shuffle() {
            int[] res = original.clone();
            Random random = new Random();
            //请认真思考下，等概率是什么。第一次被选中 * 第二次被选中的概率 = 1/n
            for (int i = 0; i < res.length; i++) {
                //[i, n) => [0, n - i) + i
                int j = random.nextInt(res.length - i) + i;
                int temp = res[i];
                res[i] = res[j];
                res[j] = temp;
            }
            return res;
        }
    }

    //[388].文件的最长绝对路径
    public static int lengthLongestPath(String input) {
        String[] split = input.split("\n");
        int res = 0;
        Stack<Integer> stack = new Stack<>();
        stack.push(0);
        for (String str : split) {
            int level = str.lastIndexOf('\t') + 1;
            while (stack.size() > level + 1) {
                stack.pop();
            }
            int cnt = stack.peek() + str.length() - level + 1;
            stack.push(cnt);
            if (str.contains(".")) {
                res = Math.max(res, cnt - 1);
            }
        }
        return res;
    }

    //[390].消除游戏
    public int lastRemaining(int n) {
        //1 2 3 4  => 2 从左到右删除剩下2
        //1 2 3 4  => 3 从右到左删除剩下3

        //1 2 3 4 5 => 2 从左到右删剩下2
        //1 2 3 4 5 => 4 从右到左删除剩下4
        //f(i) + f'(i) = i + 1

        //1 2 3 4 5 => 从左到右删除
        //2 4       => 5/2 = 2从右往左删除
        //f(i) = 2 * f'(i/2)

        //由公式可得：
        //f(i) = 2* (i/2 + 1 - f(i/2))
        return n == 1 ? 1 : 2 * (n / 2 + 1 - lastRemaining(n / 2));
    }

    //[392].判断子序列
    public boolean isSubsequence(String s, String t) {
        int i = 0, j = 0;
        while (i < s.length() && j < t.length()) {
            if (s.charAt(i) == t.charAt(j)) {
                i++;
                j++;
            }
            j++;
        }
        return i == s.length();
    }

    //[397].整数替换
    public int integerReplacement(int n) {
        int ans = 0;
        long num = n;
        //偶数肯定是除以2，奇数二进制低位出现连续的1的时候考虑+1， 只有01这种考虑-1操作，11是个特例，可以-1
        while (num != 1) {
            if (num % 2 == 0) {
                num >>= 1;
                //已经是奇数了，右移1为还是1，并且不等于3就说明了低二进制出现了可以+1的最优选择
            } else if (num != 3 && (num >> 1 & 1) == 1) {
                num++;
            } else {
                num--;
            }
            ans++;
        }
        return ans;
    }

    //[398].随机数索引
    public class Solution398 {
        int[] nums;

        public Solution398(int[] nums) {
            this.nums = nums;
        }

        public int pick(int target) {
            List<Integer> res = new ArrayList<>();
            for (int i = 0; i < nums.length; i++) {
                if (target == nums[i]) {
                    res.add(i);
                }
            }
            Random random = new Random();
            return res.get(random.nextInt(res.size()));
        }
    }

    //[400].第 N 位数字
    public static int findNthDigit(int n) {
        //i为第几层，cnt是每层的个数，length是前面几层的总位数
        long i = 1, cnt = 9, length = 0;
        for (; length + cnt * i < n; i++) {
            length += cnt * i;
            cnt *= 10;
        }
        //此时i是第几层，每个值都是i个数字，第几组就是1000 +多少就是实际数字
        long num = (long) Math.pow(10, i - 1) + (n - length - 1) / i;
        //查看在第几组的第几个数字
        int index = (int) ((n - length - 1) % i);
        return String.valueOf(num).charAt(index) - '0';
    }

    //[406].根据身高重建队列
    public int[][] reconstructQueue(int[][] people) {
        //高的不影响矮的，高的优先放置，然后再放置矮的，k值就是需要插入的索引位置
        //身高降序，位置升序
        Arrays.sort(people, (a, b) -> a[0] != b[0] ? b[0] - a[0] : a[1] - b[1]);

        List<int[]> res = new ArrayList<>();
        for (int[] p : people) {
            //按照相对位置插入即可。
            res.add(p[1], p);
        }
        return res.toArray(new int[res.size()][]);
    }

    //[419].甲板上的战舰
    public int countBattleships(char[][] board) {
        //X就改变成.，往下面和右边扩大范围
        int m = board.length, n = board[0].length;
        int ans = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] != 'X') {
                    continue;
                }
                board[i][j] = '-';
                for (int k = i + 1; k < m && board[k][j] == 'X'; k++) board[k][j] = '-';
                for (int k = j + 1; k < n && board[i][k] == 'X'; k++) board[i][k] = '-';
                ans++;
            }
        }
        return ans;
    }

    private int countBattleshipsV2(char[][] board) {
        //X就改变成.，往下面和右边扩大范围
        int m = board.length, n = board[0].length;
        int ans = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                //除了最左边和最上边的边界情况，只计算最左上角的战舰个数
                if (i > 0 && board[i - 1][j] == 'X') continue;
                if (j > 0 && board[i][j - 1] == 'X') continue;
                if (board[i][j] == 'X') {
                    ans++;
                }
            }
        }
        return ans;
    }

    //[423].从英文中重建数字
    public String originalDigits(String s) {
        //z 0, w 2, g 8, x 6, u 4, r 3, f 5, s 7, i 9, o 1
        //优先根据唯一字符干掉前面的字符，剩下的字符再取唯一字符，构建序列不唯一。
        String[] numbers = new String[]{"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
        int[] priority = new int[]{0, 2, 8, 6, 4, 3, 5, 7, 9, 1};
        int[] cnts = new int[26];
        for (char ch : s.toCharArray()) {
            cnts[ch - 'a']++;
        }
        StringBuilder sb = new StringBuilder();
        for (int p : priority) {
            //探索出来需要多少次个相同的数字，times可能为0
            int times = Integer.MAX_VALUE;
            for (char ch : numbers[p].toCharArray()) times = Math.min(times, cnts[ch - 'a']);
            if (times > 0) {
                for (char ch : numbers[p].toCharArray()) cnts[ch - 'a'] -= times;
                while (times-- > 0) sb.append(p);
            }
        }

        char[] cs = sb.toString().toCharArray();
        Arrays.sort(cs);
        return String.valueOf(cs);
    }

    //[427].建立四叉树
    public Solution427.Node construct(int[][] grid) {
        return helper(grid, 0, 0, grid.length);
    }

    private Solution427.Node helper(int[][] grid, int i, int j, int len) {
        if (len == 1) {
            return new Solution427.Node(grid[i][j] == 1, true);
        }

        int k = len / 2;
        Solution427.Node tl = helper(grid, i, j, k);
        Solution427.Node tr = helper(grid, i, j + k, k);
        Solution427.Node bl = helper(grid, i + k, j, k);
        Solution427.Node br = helper(grid, i + k, j + k, k);
        if (tl.val == tr.val && bl.val == br.val && tl.val == bl.val && tl.isLeaf && tr.isLeaf && bl.isLeaf && br.isLeaf) {
            return new Solution427.Node(tl.val, true);
        } else {
            return new Solution427.Node(true, false, tl, tr, bl, br);
        }
    }

    //[429].N 叉树的层序遍历
    public List<List<Integer>> levelOrder(Node root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;

        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> level = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                Node cur = queue.poll();
                level.add(cur.val);

                if (cur.children != null) {
                    for (Node child : cur.children) {
                        queue.offer(child);
                    }
                }
            }
            res.add(level);
        }
        return res;
    }

    //[435].无重叠区间
    public int eraseOverlapIntervals(int[][] intervals) {
        int n = intervals.length;
        int overlap = 0;
        //1,2  2,3  3,4  1,3
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        int end = intervals[0][1];
        for (int i = 1; i < n; i++) {
            int[] cur = intervals[i];
            if (cur[0] < end) {
                overlap++;
                //默认保留最短的，长的覆盖范围更大，所以删除
                end = Math.min(end, cur[1]);
            } else {
                end = cur[1];
            }
        }
        return overlap;
    }

    //[436].寻找右区间
    public int[] findRightInterval(int[][] intervals) {
        int n = intervals.length;
        int[][] sort = new int[n][2];
        //第一个索引 + 真实索引
        for (int i = 0; i < n; i++) {
            sort[i] = new int[]{intervals[i][0], i};
        }

        int[] res = new int[n];
        //保证最小排序
        Arrays.sort(sort, (a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
        //找到一个大于等于它的最小值就是左边界, labuladong模版
        for (int i = 0; i < n; i++) {
            int v = intervals[i][1];
            int left = 0, right = n - 1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (sort[mid][0] == v) {
                    right = mid - 1;
                } else if (sort[mid][0] > v) {
                    right = mid - 1;
                } else if (sort[mid][0] < v) {
                    left = mid + 1;
                }
            }
            //额外的补充条件
            res[i] = left > n - 1 || sort[left][0] < v ? -1 : sort[left][1];
        }
        return res;
    }

    //[438].找到字符串中所有字母异位词
    public List<Integer> findAnagrams(String s, String p) {
        int[] cnts = new int[26];
        int diffP = 0;
        for (char ch : p.toCharArray()) cnts[ch - 'a']++;
        for (int cnt : cnts) {
            if (cnt > 0) diffP++;
        }
        //abc
        List<Integer> res = new ArrayList<>();
        for (int l = 0, r = 0, diffA = 0; r < s.length(); r++) {
            //扣减操作，如果一个字符数量为0，说明有匹配的一个字符，需要增加一个字符计数
            if (--cnts[s.charAt(r) - 'a'] == 0) diffA++;
            if (r - l + 1 > p.length()) {
                //当恢复操作，发现一个字符变为了1，说明之前有匹配的字符，需要减掉一个字符计数
                if (++cnts[s.charAt(l++) - 'a'] == 1) diffA--;
            }
            if (diffA == diffP) res.add(l);
        }
        return res;
    }

    //[442].数组中重复的数据
    public static List<Integer> findDuplicates(int[] nums) {
        List<Integer> res = new ArrayList<>();
        //利用翻转为负数的情况来标记是否有重复，因为范围从1~n，通过num-1，就是索引的位置
        for (int num : nums) {
            //本身这个值可能被翻转过，所以要abs
            int index = Math.abs(num) - 1;

            if (nums[index] < 0) {
                //索引值+1才是真正的数字
                res.add(index + 1);
            }

            //翻转
            nums[index] = -nums[index];
        }
        return res;
    }

    //[443].压缩字符串
    public static int compress(char[] chars) {
        int index = 0;
        //需要越界，才能把最后的计算完毕
        for (int l = 0, r = 0; r <= chars.length; r++) {
            if (r == chars.length || chars[l] != chars[r]) {
                int count = r - l;
                if (count > 1) {
                    chars[index++] = chars[l];
                    for (char ch : String.valueOf(count).toCharArray()) chars[index++] = ch;
                    l = r;
                } else {
                    chars[index++] = chars[l];
                    l = r;
                }
            }
        }
        return index;
    }

    //[445].两数相加 II
    public ListNode addTwoNumbers2(ListNode l1, ListNode l2) {
        Stack<Integer> s1 = new Stack<>();
        Stack<Integer> s2 = new Stack<>();
        while (l1 != null) {
            s1.push(l1.val);
            l1 = l1.next;
        }
        while (l2 != null) {
            s2.push(l2.val);
            l2 = l2.next;
        }

        int carry = 0;
        ListNode dummy = new ListNode(-1);
        while (!s1.isEmpty() || !s2.isEmpty() || carry != 0) {
            int x = s1.isEmpty() ? 0 : s1.pop();
            int y = s2.isEmpty() ? 0 : s2.pop();
            int sum = x + y + carry;
            ListNode cur = new ListNode(sum % 10);
            carry = sum / 10;
            cur.next = dummy.next;
            dummy.next = cur;
        }
        return dummy.next;
    }

    //[449].序列化和反序列化二叉搜索树
    public class Codec2 {

        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            if (root == null) return "";
            StringBuilder sb = new StringBuilder();
            dfsForSerialize(root, sb);
            return sb.deleteCharAt(sb.length() - 1).toString();
        }

        private void dfsForSerialize(TreeNode root, StringBuilder sb) {
            if (root == null) return;
            sb.append(root.val).append(',');
            dfsForSerialize(root.left, sb);
            dfsForSerialize(root.right, sb);
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            if (data == null || data.length() == 0) return null;
            String[] items = data.split(",");
            return dfsForDeserialize(items, 0, items.length - 1);
        }

        private TreeNode dfsForDeserialize(String[] list, int start, int end) {
            if (start > end) return null;
            String val = list[start];
            TreeNode root = new TreeNode(Integer.parseInt(val));

            int index = end + 1;
            for (int i = start + 1; i <= end; i++) {
                if (Integer.parseInt(list[i]) > Integer.parseInt(val)) {
                    index = i;
                    break;
                }
            }
            root.left = dfsForDeserialize(list, start + 1, index - 1);
            root.right = dfsForDeserialize(list, index, end);
            return root;
        }
    }

    //[450].删除二叉搜索树中的节点
    public TreeNode deleteNode(TreeNode root, int key) {
        //可能搜不到，直到搜到叶子节点，直接返回
        if (root == null) return null;
        if (root.val < key) {
            root.right = deleteNode(root.right, key);
        } else if (root.val > key) {
            root.left = deleteNode(root.left, key);
        } else {
            //返回的是新的头结点，因为是叶子节点，直接返回null为新的头
            if (root.left == null && root.right == null) {
                return null;
            }
            if (root.left == null) return root.right;
            if (root.right == null) return root.left;

            TreeNode rightMin = getRightMin(root.right);
            root.right = deleteNode(root.right, rightMin.val);
            rightMin.left = root.left;
            rightMin.right = root.right;
            root = rightMin;
        }
        return root;
    }

    private TreeNode getRightMin(TreeNode right) {
        TreeNode cur = right;
        while (cur.left != null) {
            cur = cur.left;
        }
        return cur;
    }

    //[456].132 模式
    public boolean find132pattern(int[] nums) {
        int n = nums.length;
        if (n <= 2) return false;
        //[3,5,0,3,4]
        Stack<Integer> stack = new Stack<>();
        //维护是一个单调递减栈
        int maxRight = Integer.MIN_VALUE;
        for (int i = n - 1; i >= 0; i--) {
            if (nums[i] < maxRight) {
                return true;
            }
            while (!stack.isEmpty() && nums[i] > stack.peek()) {
                maxRight = Math.max(maxRight, stack.pop());
            }
            stack.push(nums[i]);
        }
        return false;
    }

    //[463].岛屿的周长
    public int islandPerimeter(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        //1是岛屿，0是水
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    //题目说了只有一个
                    return dfsForIslandPerimeter(grid, i, j);
                }
            }
        }
        return 0;
    }

    private int dfsForIslandPerimeter(int[][] grid, int x, int y) {
        int m = grid.length, n = grid[0].length;
        //遇到边界就+1
        if (x < 0 || y < 0 || x >= m || y >= n) {
            return 1;
        }
        //遇到水域+1
        if (grid[x][y] == 0) {
            return 1;
        }
        //已经遍历过了，计数为0
        if (grid[x][y] == 2) {
            return 0;
        }
        grid[x][y] = 2;
        return dfsForIslandPerimeter(grid, x - 1, y)
                + dfsForIslandPerimeter(grid, x + 1, y)
                + dfsForIslandPerimeter(grid, x, y - 1)
                + dfsForIslandPerimeter(grid, x, y + 1);
    }

    //[468].验证IP地址
    public String validIPAddress(String queryIP) {
        if (queryIP.contains(".")) {
            String[] ipv4 = queryIP.split("\\.", -1);
            if (ipv4.length != 4) {
                return "Neither";
            }
            for (String ip : ipv4) {
                if (ip.length() > 3 || ip.length() <= 0) {
                    return "Neither";
                }
                for (char ch : ip.toCharArray()) {
                    if (!Character.isDigit(ch)) {
                        return "Neither";
                    }
                }
                int num = Integer.parseInt(ip);
                if (num > 255 || num < 0 || String.valueOf(num).length() != ip.length()) {
                    return "Neither";
                }
            }
            return "IPv4";
        } else if (queryIP.contains(":")) {
            String[] ipv6 = queryIP.split(":", -1);
            if (ipv6.length != 8) {
                return "Neither";
            }
            for (String ip : ipv6) {
                if (ip.length() > 4 || ip.length() <= 0) {
                    return "Neither";
                }
                for (char ch : ip.toCharArray()) {
                    if (!Character.isDigit(ch) && !('a' <= ch && ch <= 'f') && !('A' <= ch && ch <= 'F')) {
                        return "Neither";
                    }
                }
            }
            return "IPv6";
        }
        return "Neither";
    }

    //[470].用 Rand7() 实现 Rand10()
    class Solution470 {
        //虚拟方法
        int rand7() {
            return 0;
        }

        public int rand10() {
            while (true) {
                //转化为7进制，覆盖范围为[0, 48]
                int rand = (rand7() - 1) * 7 + (rand7() - 1);
                //扩大为倍数，每个数都是等概率的，然后扩大到[1,10]的最大的倍数4，较少rand7的调用次数
                if (1 <= rand && rand <= 40)
                    //rand % 10 取值范围是[0,9] + 1 => [1, 10]
                    return rand % 10 + 1;
            }
        }
    }

    //[472].连接词
    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        TrieNode trie = new TrieNode();
        List<String> res = new ArrayList<>();
        //首先是短的在前面，长的在后面，用长的去匹配短，如果能到结尾，说明是成功的，否则是不成功的。
        Arrays.sort(words, Comparator.comparingInt(String::length));
        //一个单词的匹配需要从头到尾逐一匹配前面的单词，所以有选择，避免不了用深度优先遍历
        for (String word : words) {
            if (word.length() == 0) continue;
            if (dfsForFindAllConcatenateWordsInADict(trie, word, 0)) {
                res.add(word);
            } else {
                insertToTrie(trie, word);
            }
        }
        return res;
    }

    private boolean dfsForFindAllConcatenateWordsInADict(TrieNode root, String word, int start) {
        if (start == word.length()) return true;
        TrieNode cur = root;
        //这么多选择
        for (int i = start; i < word.length(); i++) {
            int ch = word.charAt(i) - 'a';
            cur = cur.children[ch];
            if (cur == null) {
                return false;
            }
            //一定要终结了才可以选择下一个位置
            if (cur.isEnd) {
                //传root，从头开始匹配
                if (dfsForFindAllConcatenateWordsInADict(root, word, i + 1)) {
                    return true;
                }
            }
        }
        return false;
    }

    private void insertToTrie(TrieNode root, String word) {
        TrieNode cur = root;
        for (int i = 0; i < word.length(); i++) {
            int ch = word.charAt(i) - 'a';
            if (cur.children[ch] == null) {
                cur.children[ch] = new TrieNode();
            }
            cur = cur.children[ch];
        }
        cur.isEnd = true;
    }

    //[474].一和零
    public int findMaxForm(String[] strs, int m, int n) {
        //dp[k][i][j] 前k的物品，0的容量为i，1的容量为j的最大子集个数
//        int s = strs.length;
//        int[][][] dp = new int[s + 1][m + 1][n + 1];
//        for (int k = 1; k <= s; k++) {
//            String str = strs[k - 1];
//            int zero = 0, one = 0;
//            for (int z = 0; z < str.length(); z++) {
//                if (str.charAt(z) == '0') zero++;
//                else one++;
//            }
//
//            for (int i = 0; i <= m; i++) {
//                for (int j = 0; j <= n; j++) {
//                    //容量够，选择 和 不选择
//                    if (i >= zero && j >= one) {
//                        //选择，更新子集数+1
//                        dp[k][i][j] = Math.max(dp[k - 1][i][j], dp[k - 1][i - zero][j - one] + 1);
//                    } else {
//                        //太小装不下
//                        dp[k][i][j] = dp[k-1][i][j];
//                    }
//                }
//            }
//        }
//        return dp[s][m][n];

        //因为dp[k][i][j] 依赖于 dp[k-1][i][j]和dp[k - 1][i - zero][j - one]，一个左上方，一个正上方，而dp[s][m][n]就是答案
        //压缩掉物品维度后，dp[i][j] 依赖上一次的dp[i][j]和dp[i-zero][j-one]，
        //如果正序遍历，dp[i-zero][j-one]会影响dp[i][j]结果，就是前面的结果造成了后面结果的覆盖，会导致重复计算。
        //如果倒序遍历，就避免了滚动数组的覆盖问题。
        int s = strs.length;
        int[][] dp = new int[m + 1][n + 1];
        for (int k = 1; k <= s; k++) {
            String str = strs[k - 1];
            int zero = 0, one = 0;
            for (int z = 0; z < str.length(); z++) {
                if (str.charAt(z) == '0') zero++;
                else one++;
            }

            for (int i = m; i >= 0; i--) {
                for (int j = n; j >= 0; j--) {
                    //容量够，选择 和 不选择
                    if (i >= zero && j >= one) {
                        //选择，更新子集数+1
                        dp[i][j] = Math.max(dp[i][j], dp[i - zero][j - one] + 1);
                    } else {
                        //太小装不下
                        dp[i][j] = dp[i][j];
                    }
                }
            }
        }
        return dp[m][n];
    }

    //[475].供暖器
    public static int findRadius(int[] houses, int[] heaters) {
        Arrays.sort(heaters);
        int ans = 0, n = heaters.length;
        for (int house : houses) {
            int left = findRightIndex(heaters, house, n);
            int right = left + 1;
            int leftRadius = left < 0 ? Integer.MAX_VALUE : house - heaters[left];
            int rightRadius = right >= n ? Integer.MAX_VALUE : heaters[right] - house;
            ans = Math.max(ans, Math.min(leftRadius, rightRadius));
        }
        return ans;
    }

    private static int findRightIndex(int[] heaters, int house, int n) {
        //1,2,4
        //找到位置小于等于house的heater
        //如果没有相等的，那么就找到小于它的，如果有相等的，找它自己
        int left = 0, right = n - 1;
        while (left < right) {
            int mid = left + (right - left + 1) / 2;
            if (heaters[mid] <= house) {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        if (heaters[left] > house) {
            return -1;
        }
        return left;
    }

    //[478].在圆内随机生成点
    public class Solution478 {
        double rad;
        double x;
        double y;

        public Solution478(double radius, double x_center, double y_center) {
            rad = radius;
            x = x_center;
            y = y_center;
        }

        public double[] randPoint() {
            //左下角的位置为原始点
            double x0 = x - rad;
            double y0 = y - rad;
            while (true) {
                //分别取x，y的随机点，原始点 + 2 * radius * [0, 1]随机值
                double nx = x0 + Math.random() * 2 * rad;
                double ny = y0 + Math.random() * 2 * rad;
                //如果采样不在圈内，表示概率不相等继续选择
                if (Math.sqrt(Math.pow(nx - x, 2) + Math.pow(ny - y, 2)) <= rad) {
                    return new double[]{nx, ny};
                }
            }
        }
    }

    //[491].递增子序列
    public List<List<Integer>> findSubsequences(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        backtraceForFindSubsequences(nums, 0, res, new LinkedList<>());
        return res;
    }

    private void backtraceForFindSubsequences(int[] nums, int s, List<List<Integer>> res, LinkedList<Integer> select) {
        if (select.size() >= 2) {
            res.add(new ArrayList<>(select));
        }

        Set<Integer> set = new HashSet<>();
        for (int i = s; i < nums.length; i++) {
            if (set.contains(nums[i])) {
                continue;
            }
            if (select.size() > 0 && nums[i] < select.getLast()) {
                continue;
            }

            set.add(nums[i]);
            select.addLast(nums[i]);
            backtraceForFindSubsequences(nums, i + 1, res, select);
            select.removeLast();
        }
    }

    //[495].提莫攻击
    public int findPoisonedDuration(int[] timeSeries, int duration) {
        //注意第0秒需要考虑到， last标记为上一次攻击的结束时间
        //如果last < s 没重合，+duration；如果last >=s 有重合，+ e - last
        int ans = 0, last = -1;
        for (int s : timeSeries) {
            int e = s + duration - 1;
            ans += last < s ? duration : e - last;
            last = e;
        }
        return ans;
    }

    //[496].下一个更大元素I
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Map<Integer, Integer> numberMap = new HashMap<>();
        Stack<Integer> stack = new Stack<>();
        for (int i = nums2.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && nums2[i] >= nums2[stack.peek()]) {
                stack.pop();
            }
            numberMap.put(nums2[i], stack.isEmpty() ? -1 : nums2[stack.peek()]);
            stack.push(i);
        }

        int[] res = new int[nums1.length];
        for (int i = 0; i < nums1.length; i++) {
            res[i] = numberMap.get(nums1[i]);
        }
        return res;
    }

    //[498].对角线遍历
    public int[] findDiagonalOrder(int[][] mat) {
        // 00 -> 0,1 -> 1,0 -> 2,0 -> 1,1 -> 0,2 -> 1,2 -> 2,1 -> 3,0
        //x < 0   x = 0 y = y + 1
        //y >= n  x = x+1, y = n-1
        //y < 0   x=  x+1, y = 0
        //x >= n  x = n-1,  y = y+1;
        int m = mat.length, n = mat[0].length;
        int r = 0, c = 0;
        int[] res = new int[m * n];
        for (int i = 0; i < res.length; i++) {
            res[i] = mat[r][c];
            //偶数往上
            if ((r + c) % 2 == 0) {
                //优先判断是不是右边界超了，（上边界和右边界可能同时超，往下边走）
                if (c == n - 1) {
                    r++;
                } else if (r == 0) {
                    c++;
                } else {
                    r--;
                    c++;
                }
            } else {
                //奇数往下
                //优先判断是不是下边界超了，（下边界和左边界可能同时超，往后边走）
                if (r == m - 1) {
                    c++;
                } else if (c == 0) {
                    r++;
                } else {
                    r++;
                    c--;
                }
            }
        }
        return res;
    }

    //[503].下一个更大的元素II
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        Stack<Integer> stack = new Stack<>();
        for (int i = 2 * n - 1; i >= 0; i--) {
            while (!stack.isEmpty() && nums[i % n] >= stack.peek()) {
                stack.pop();
            }

            res[i % n] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(nums[i % n]);
        }
        return res;
    }

    //[506].相对名次
    public String[] findRelativeRanks(int[] score) {
        int n = score.length;
        int[] clone = score.clone();
        Arrays.sort(clone);
        Map<Integer, Integer> order = new HashMap<>();
        for (int i = 1; i <= n; i++) {
            order.put(clone[i - 1], n - i + 1);
        }

        String[] ans = new String[n];
        for (int i = 0; i < n; i++) {
            int no = order.get(score[i]);
            if (no == 1) {
                ans[i] = "Gold Medal";
            } else if (no == 2) {
                ans[i] = "Silver Medal";
            } else if (no == 3) {
                ans[i] = "Bronze Medal";
            } else {
                ans[i] = String.valueOf(no);
            }
        }
        return ans;
    }

    //[507].完美数
    public boolean checkPerfectNumber(int num) {
        if (num == 1) return false;
        int sum = 1;
        //36以6为界，左边从2到6，就是右边18到6，计算的时候1排除在外
        //1 2 3 4 6 9 12 18
        for (int i = 2; i <= num / i; i++) {
            if (num % i == 0) {
                sum += i;
                sum += num / i;
            }
        }
        return sum == num;
    }

    //[509].斐波那契数
    public int fib(int n) {
//        int[] dp = new int[n + 1];
//        dp[0] = 0;
//        dp[1] = 1;
//        for (int i = 2; i <= n; i++) {
//            dp[i] = dp[i - 1] + dp[i - 2];
//        }
//        return dp[n];
        int dp_i_2 = 0;
        int dp_i_1 = 1;
        int dp_i = 0;
        for (int i = 2; i <= n; i++) {
            dp_i = dp_i_2 + dp_i_1;
            dp_i_2 = dp_i_1;
            dp_i_1 = dp_i;
        }
        return dp_i;
    }

    //[513].找树左下角的值
    private int maxDepth = -1, leftValue = -1;

    public int findBottomLeftValue(TreeNode root) {
        dfsForFindBottomLeftValue(root, 0);
        return leftValue;
    }

    private void dfsForFindBottomLeftValue(TreeNode root, int depth) {
        if (root == null) return;
        if (root.left == null && root.right == null) {
            if (depth > maxDepth) {
                leftValue = root.val;
                maxDepth = depth;
            }
        }
        dfsForFindBottomLeftValue(root.left, depth + 1);
        dfsForFindBottomLeftValue(root.right, depth + 1);
    }

    //[515].在每个树行中找最大值
    private List<Integer> res = new ArrayList<>();

    public List<Integer> largestValues(TreeNode root) {
        dfsForLargestValues(root, 0);
        return res;
    }

    private void dfsForLargestValues(TreeNode root, int depth) {
        if (root == null) return;
        if (res.size() == depth) {
            res.add(root.val);
        } else {
            int cur = res.get(depth);
            res.set(depth, Math.max(cur, root.val));
        }

        dfsForLargestValues(root.left, depth + 1);
        dfsForLargestValues(root.right, depth + 1);
    }

    //[519].随机翻转矩阵
    public class Solution519 {
        //首先这个题目需要优化空间，二维转一维，
        //每次随机之后会被占用，每次占用之后用最后的位置代替可选位置，可以保证实际位置的连续
        int row, col, n;
        //实际可以用的位置，默认i->i，每当一个i被占用，就和最后一个位置交换下
        Map<Integer, Integer> index;
        Random random = new Random();

        public Solution519(int m, int n) {
            this.row = m;
            this.col = n;
            this.n = m * n;
            this.index = new HashMap<>();
        }

        public int[] flip() {
            //[0,n)位置选择
            int r = random.nextInt(n--);

            int idx = index.getOrDefault(r, r);

            //意思是 我占用了该位置，可以替换的新位置是n-1(最后的索引位置)
            //跟一维数组交换最后位置一样
            index.put(idx, index.getOrDefault(n, n));
            return new int[]{idx / col, idx % col};
        }

        public void reset() {
            index.clear();
            n = row * col;
        }
    }

    //[523].连续的子数组和
    public boolean checkSubarraySum(int[] nums, int k) {
        if (nums.length < 2) return false;

        //[23,2,4,6,7], k = 6
        //子数组和，需要的是快速定位区间和，另外需要判断连续子数组，已知一个值和结果，那么就可以用hash来判断，跟两数之和类似
        //k的倍数， (preSum[j] - preSum[i]) % k == 0; 即前后两个前缀和取余数相等就是判断条件！！！
        Map<Integer, Integer> preSumMap = new HashMap<>();
        preSumMap.put(0, -1);
        int preSum = 0;
        for (int i = 0; i < nums.length; i++) {
            preSum += nums[i];

            int reminder = preSum % k;
            if (preSumMap.containsKey(reminder)) {
                if (i - preSumMap.get(reminder) > 1) {
                    return true;
                }
            } else {
                preSumMap.put(reminder, i);
            }
        }
        return false;
    }

    //[525].连续数组
    public int findMaxLength(int[] nums) {
        //首先求解是连续数组，所以需要构造区间统计0和1的个数，可以0的时候前缀和-1， 1的时候前缀和+1
        //当某个前缀和再次出现的时候，一定代表0和1是相等的。
        Map<Integer, Integer> preSumMap = new HashMap<>();
        preSumMap.put(0, -1);
        int preSum = 0, res = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 1) {
                preSum++;
            } else {
                preSum--;
            }
            if (preSumMap.containsKey(preSum)) {
                res = Math.max(res, i - preSumMap.get(preSum));
            } else {
                preSumMap.put(preSum, i);
            }
        }
        return res;
    }

    //[528].按权重随机选择
    public static class Solution528 {
        int[] preSum;
        Random random = new Random();

        public Solution528(int[] w) {
            int n = w.length;
            preSum = new int[n];
            preSum[0] = w[0];
            for (int i = 1; i < n; i++) {
                preSum[i] = preSum[i - 1] + w[i];
            }
        }

        public int pickIndex() {
            int n = preSum.length;
            //选择[1, total]之间的值
            int w = random.nextInt(preSum[n - 1]) + 1;
            //左边界算法
            int left = 0, right = n - 1;
            while (left < right) {
                int mid = left + (right - left) / 2;
                if (preSum[mid] >= w) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            return left;
        }
    }

    //[535].TinyURL 的加密与解密
    public class Codec535 {
        String encode = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        Map<String, String> shortLong = new HashMap<>();
        Random random = new Random();

        // Encodes a URL to a shortened URL.
        public String encode(String longUrl) {
            String key = getKey();
            while (shortLong.containsKey(key)) {
                key = getKey();
            }

            shortLong.put(key, longUrl);
            return "http://tinyurl.com/" + key;
        }

        // Decodes a shortened URL to its original URL.
        public String decode(String shortUrl) {
            String key = shortUrl.replace("http://tinyurl.com/", "");
            return shortLong.get(key);
        }

        private String getKey() {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 6; i++) {
                sb.append(encode.charAt(random.nextInt(encode.length())));
            }
            return sb.toString();
        }
    }

    //[537].复数乘法
    public String complexNumberMultiply(String num1, String num2) {
        String[] sp1 = num1.split("\\+");
        String[] sp2 = num2.split("\\+");
        int s1 = Integer.parseInt(sp1[0]);
        int x1 = Integer.parseInt(sp1[1].replace("i", ""));

        int s2 = Integer.parseInt(sp2[0]);
        int x2 = Integer.parseInt(sp2[1].replace("i", ""));
        return s1 * s2 - x1 * x2 + "+" + (s1 * x2 + s2 * x1) + "i";
    }

    //[538].把二叉搜索树转换为累加树
    //[1038].把二叉搜索树转换为累加树
    public TreeNode convertBST(TreeNode root) {
        //右根左这样的递归顺序遍历
        dfsForConvertBST(root, new AtomicInteger());
        return root;
    }

    private void dfsForConvertBST(TreeNode root, AtomicInteger sum) {
        if (root == null) return;
        dfsForConvertBST(root.right, sum);
        sum.addAndGet(root.val);

        root.val = sum.get();
        dfsForConvertBST(root.left, sum);
    }

    //[539].最小时间差
    public int findMinDifference(List<String> timePoints) {
        //抽屉原理，总时间为1440，超过必然有一个重复的
        if (timePoints.size() > 1440) return 0;
        int[] cnts = new int[1440 * 2];
        for (String timePoint : timePoints) {
            String[] vals = timePoint.split(":");
            int time = Integer.parseInt(vals[0]) * 60 + Integer.parseInt(vals[1]);
            cnts[time]++;
            //将最小的时间扩展到下一天
            cnts[time + 1440]++;
        }
        int ans = 1440, idx = -1;
        for (int i = 0; i < 1440 * 2 && ans != 0; i++) {
            if (cnts[i] == 0) continue;
            if (cnts[i] > 1) return 0;
            //相同间隔逐一比较，尤其是数量为1的情况
            if (idx != -1) ans = Math.min(ans, i - idx);
            idx = i;
        }
        return ans;
    }

    //[540].有序数组中的单一元素
    public int singleNonDuplicate(int[] nums) {
        int n = nums.length;
        int l = 0, r = n - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            boolean halfOdd = ((r - mid) % 2 == 0);
            if (nums[mid - 1] == nums[mid]) {
                if (halfOdd) {
                    r = mid - 2;
                } else {
                    l = mid + 1;
                }
            } else if (nums[mid + 1] == nums[mid]) {
                if (halfOdd) {
                    l = mid + 2;
                } else {
                    r = mid - 1;
                }
            } else {
                return nums[mid];
            }
        }
        return nums[l];
    }

    //[542].01 矩阵
    public int[][] updateMatrix(int[][] mat) {
        int m = mat.length, n = mat[0].length;
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                //从0开始，逐层往外面扩，扩的同时，必须访问未被访问过的节点，不然数据更新会有问题
                if (mat[i][j] == 0) {
                    queue.offer(new int[]{i, j});
                } else {
                    //标记未被访问过
                    mat[i][j] = -1;
                }
            }
        }

        int[][] direct = new int[][]{{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0], y = cur[1];
            for (int[] dir : direct) {
                int newX = x + dir[0];
                int newY = y + dir[1];
                //未被访问过，则更新
                if (newX >= 0 && newX < m && newY >= 0 && newY < n && mat[newX][newY] == -1) {
                    mat[newX][newY] = mat[x][y] + 1;
                    queue.offer(new int[]{newX, newY});
                }
            }
        }
        return mat;
    }

    //[547].省份数量
    public int findCircleNum(int[][] isConnected) {
        int n = isConnected.length;
        UnionFind uf = new UnionFind(n);
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (isConnected[i][j] == 1) {
                    uf.union(i, j);
                }
            }
        }
        return uf.count;
    }

    //[559].N 叉树的最大深度
    public int maxDepth(Node root) {
        if (root == null) return 0;
        int maxChildDepth = 0;
        for (Node child : root.children) {
            maxChildDepth = Math.max(maxDepth(child), maxChildDepth);
        }
        return maxChildDepth + 1;
    }

    //[560].和为 K 的子数组
    public int subarraySum(int[] nums, int k) {
        //[1,1,1]
        int preSum = 0, res = 0;
        Map<Integer, Integer> preSumCount = new HashMap<>();
        //当只有一个元素满足k的时候，需要统计为1
        preSumCount.put(0, 1);
        for (int i = 0; i < nums.length; i++) {
            preSum += nums[i];
            //preSum比k大
            if (preSumCount.containsKey(preSum - k)) {
                res += preSumCount.get(preSum - k);
            }
            preSumCount.put(preSum, preSumCount.getOrDefault(preSum, 0) + 1);
        }
        return res;
    }

    //[581].最短无序连续子数组
    public int findUnsortedSubarray(int[] nums) {
        int right = 0;
        int left = nums.length - 1;
        Stack<Integer> stack = new Stack<>();
        for (int i = nums.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && nums[i] > nums[stack.peek()]) {
                right = Math.max(right, stack.pop());
            }
            stack.push(i);
        }

        //一个栈没办法同时找到左右边界
        stack.clear();
        for (int i = 0; i < nums.length; i++) {
            while (!stack.isEmpty() && nums[i] < nums[stack.peek()]) {
                left = Math.min(left, stack.pop());
            }
            stack.push(i);
        }

        return right > left ? right - left + 1 : 0;
    }

    //[583].两个字符串的删除操作
    public int minDistance(String word1, String word2) {
//        int common = longestCommonSubsequence(word1, word2);
//        return word1.length() + word2.length() - 2 * common;

        int m = word1.length(), n = word2.length();
        //字符串，定义为长度为i的word1，长度为j的word2的最小删除操作数
        int[][] dp = new int[m + 1][n + 1];
        //长度为0，另一个的编辑距离递增
        for (int i = 0; i <= m; i++) dp[i][0] = i;
        for (int i = 0; i <= n; i++) dp[0][i] = i;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    //不相等，砍掉两个字符删除数就+2，砍掉一个字符删除数就+1。
                    dp[i][j] = Math.min(dp[i - 1][j - 1] + 2, Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1));
                }
            }
        }
        return dp[m][n];
    }

    //[589].N 叉树的前序遍历
    public List<Integer> preorder(Node root) {
        Stack<Node> stack = new Stack<>();
        List<Integer> res = new ArrayList<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            Node cur = stack.pop();
            res.add(cur.val);
            if (cur.children != null) {
                for (int i = cur.children.size() - 1; i >= 0; i--) {
                    stack.push(cur.children.get(i));
                }
            }
        }
        return res;
    }

    //[590].N 叉树的后序遍历
    public List<Integer> postorder(Node root) {
        Stack<Node> stack = new Stack<>();
        LinkedList<Integer> res = new LinkedList<>();
        stack.add(root);
        //根右左 倒序插入就是左右根
        while (!stack.isEmpty()) {
            Node cur = stack.pop();
            res.addFirst(cur.val);
            //从左到右依次插入，出栈顺序就是从右到左
            for (Node child : cur.children) {
                if (child != null) {
                    stack.push(child);
                }
            }
        }
        return res;
    }

    //[592].分数加减运算
    public String fractionAddition(String expression) {
        //第一位-号不需要替换，不然麻烦
        String[] split = (expression.substring(0, 1) + expression.substring(1).replace("-", "+-")).split("\\+");
        String cur = split[0];
        for (int i = 1; i < split.length; i++) {
            String next = split[i];
            String[] curArr = cur.split("/");
            String[] nextArr = next.split("/");

            int mother = lcm(Integer.parseInt(curArr[1]), Integer.parseInt(nextArr[1]));
            int son = Integer.parseInt(curArr[0]) * mother / Integer.parseInt(curArr[1]) + Integer.parseInt(nextArr[0]) * mother / Integer.parseInt(nextArr[1]);
            cur = son + "/" + mother;
        }
        int son = Integer.parseInt(cur.split("/")[0]);
        int mother = Integer.parseInt(cur.split("/")[1]);
        //最大公约数
        int g = Math.abs(gcd(son, mother));
        return son / g + "/" + mother / g;
    }

    private int gcd(int a, int b) {
        //辗转相除法求最大公约数
        while (b != 0) {
            int t = b;
            b = a % b;
            a = t;
        }
        return a;
    }

    private int lcm(int a, int b) {
        return a * b / gcd(a, b);
    }

    //[593].有效的正方形
    public boolean validSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
        Set<Integer> set = new HashSet<>();
        //计算两两之间的距离，如果是正三角形和重心，会出现两种距离，但是都是整数，所以重心不可能出现。
        set.add(calDistance(p1, p2));
        set.add(calDistance(p1, p3));
        set.add(calDistance(p1, p4));
        set.add(calDistance(p2, p3));
        set.add(calDistance(p2, p4));
        set.add(calDistance(p3, p4));
        return set.size() == 2 && !set.contains(0);
    }

    private int calDistance(int[] a, int[] b) {
        return (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]);
    }

    //[598].范围求和 II
    public int maxCount(int m, int n, int[][] ops) {
        int minM = m, minN = n;
        for (int[] p : ops) {
            minM = Math.min(p[0], minM);
            minN = Math.min(p[1], minN);
        }
        return minM * minN;
    }

    //[611].有效三角形的个数
    public int triangleNumber(int[] nums) {
        int n = nums.length;
        if (n < 3) return 0;

        Arrays.sort(nums);

        int res = 0;
        for (int i = n - 1; i >= 2; i--) {
            //l最小值，r次大值
            int l = 0, r = i - 1;
            while (l < r) {
                //以r次大，找到最小l的边界；然后再缩小r，找最小l的边界
                if (nums[l] + nums[r] > nums[i]) {
                    //更新个数
                    res += r - l;
                    r--;
                } else {
                    l++;
                }
            }
        }
        return res;
    }

    //[623].在二叉树中增加一行
    public TreeNode addOneRow(TreeNode root, int val, int depth) {
        //细节点1
        if (depth == 1) {
            TreeNode n = new TreeNode(val);
            n.left = root;
            return n;
        }
        helperForAddOneRow(root, val, depth, 1);
        return root;
    }

    private void helperForAddOneRow(TreeNode root, int val, int depth, int level) {
        if (root == null) return;
        if (level == depth - 1) {
            //细节点2
            TreeNode left = new TreeNode(val);
            left.left = root.left;
            root.left = left;

            TreeNode right = new TreeNode(val);
            right.right = root.right;
            root.right = right;
            return;
        }

        helperForAddOneRow(root.left, val, depth, level + 1);
        helperForAddOneRow(root.right, val, depth, level + 1);
    }

    //[630].课程表 III
    public int scheduleCourse(int[][] courses) {
        //将课程按照最后的截止日期从小到大，起始时间从小到大排序
        Arrays.sort(courses, (a, b) -> a[1] == b[1] ? a[0] - b[0] : a[1] - b[1]);
        //大根堆，如果当前课程可以正常结束，则更新结束时间; 如果当前课程不可以正常结束，替换掉之前持续时间比它大的已经选入的课程，并更新结束时间
        PriorityQueue<Integer> queue = new PriorityQueue<>((a, b) -> b - a);
        int finishAt = 0;
        for (int[] course : courses) {
            if (finishAt + course[0] <= course[1]) {
                finishAt += course[0];
                queue.offer(course[0]);
            } else if (!queue.isEmpty() && queue.peek() > course[0]) {
                finishAt -= queue.poll() - course[0];
                queue.offer(course[0]);
            }
        }
        return queue.size();
    }

    //[633].平方数之和
    public boolean judgeSquareSum(int c) {
        int left = 0, right = (int) Math.sqrt(c);
        while (left <= right) {
            int sum = left * left + right * right;
            if (sum == c) {
                return true;
            } else if (sum < c) {
                left++;
            } else {
                right--;
            }
        }
        return false;
    }

    //[674].最长连续递增序列
    public int findLengthOfLCIS(int[] nums) {
        //1 2 5 4 7
        int n = nums.length;
        if (n == 1) return 1;
        int left = 0, right = 1, res = 0;
        while (right < n) {
            if (nums[right - 1] >= nums[right]) {
                left = right;
            }
            res = Math.max(res, right - left + 1);
            right++;
        }
        return res;
    }

    //[686].重复叠加字符串匹配
    public static int repeatedStringMatch(String a, String b) {
        StringBuilder sb = new StringBuilder();
        //最多遍历 (b/a + 2)a => b + 2a 长度
        int res = 0;
        while (sb.length() < b.length() + 2 * a.length()) {
            sb.append(a);
            res++;
            if (sb.toString().contains(b)) {
                return res;
            }
        }
        return -1;
    }

    //[689].三个无重叠子数组的最大和
    public int[] maxSumOfThreeSubarrays(int[] nums, int k) {
        int[] res = new int[3];
        int sum1 = 0, maxSum1 = 0, maxIdx1 = 0;
        int sum2 = 0, maxSum12 = 0, maxIdx21 = 0, maxIdx22 = 0;
        int sum3 = 0, maxSum123 = 0;

        for (int i = 2 * k; i < nums.length; i++) {
            sum1 += nums[i - 2 * k];
            sum2 += nums[i - k];
            sum3 += nums[i];
            if (i >= 3 * k - 1) {
                if (sum1 > maxSum1) {
                    maxSum1 = sum1;
                    maxIdx1 = i - 3 * k + 1;
                }
                if (sum2 + maxSum1 > maxSum12) {
                    maxSum12 = sum2 + maxSum1;
                    maxIdx21 = maxIdx1;
                    maxIdx22 = i - 2 * k + 1;
                }
                if (sum3 + maxSum12 > maxSum123) {
                    maxSum123 = sum3 + maxSum12;
                    res[0] = maxIdx21;
                    res[1] = maxIdx22;
                    res[2] = i - k + 1;
                }

                sum1 -= nums[i - 3 * k + 1];
                sum2 -= nums[i - 2 * k + 1];
                sum3 -= nums[i - k + 1];
            }
        }
        return res;
    }

    //[695].岛屿的最大面积
    public int maxAreaOfIsland(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    int area = dfsForMaxAreaOfIsland(grid, i, j);
                    res = Math.max(res, area);
                }
            }
        }
        return res;
    }

    private int dfsForMaxAreaOfIsland(int[][] grid, int x, int y) {
        int m = grid.length, n = grid[0].length;
        if (x < 0 || y < 0 || x >= m || y >= n) {
            return 0;
        }
        if (grid[x][y] == 0) {
            return 0;
        }
        grid[x][y] = 0;
        return dfsForMaxAreaOfIsland(grid, x - 1, y)
                + dfsForMaxAreaOfIsland(grid, x + 1, y)
                + dfsForMaxAreaOfIsland(grid, x, y - 1)
                + dfsForMaxAreaOfIsland(grid, x, y + 1) + 1;
    }

    //[700].二叉搜索树中的搜索
    public TreeNode searchBST(TreeNode root, int val) {
        if (root == null) return null;
        while (root != null) {
            if (root.val == val) {
                return root;
            }
            root = root.val < val ? root.right : root.left;
        }
        return null;
    }

    //[701].二叉搜索树中的插入操作
    public TreeNode insertIntoBST(TreeNode root, int val) {
//        TreeNode parent = null;
//        while (root != null) {
//            parent = root;
//            if (root.val < val) {
//                root = root.right;
//            } else {
//                root = root.left;
//            }
//        }
//        if (parent.val < val) {
//            parent.right = new TreeNode(val);
//        } else {
//            parent.left = new TreeNode(val);
//        }
//        return root;
        //递归解法
        //返回插入的头结点之后由上一次拼接完成
        if (root == null) return new TreeNode(val);
        if (root.val < val) root.right = insertIntoBST(root.right, val);
        else root.left = insertIntoBST(root.left, val);
        //返回头结点
        return root;
    }

    //[704].二分查找
    public int search2(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return -1;
    }

    //[709].转换成小写字母
    public String toLowerCase(String s) {
        StringBuilder sb = new StringBuilder();
        char[] chars = s.toCharArray();
        for (char ch : chars) {
            if (ch >= 'A' && ch <= 'Z') {
                ch |= 32;
            }
            sb.append(ch);
        }
        return sb.toString();
    }

    //[712].两个字符串的最小ASCII删除和
    public int minimumDeleteSum(String s1, String s2) {
        //求解最小ASCII删除和，可以转化为求最大公共子序列的和
        int m = s1.length(), n = s2.length();
        //长度为i的s1和长度为j的s2的最大公共子序列和
        int[][] dp = new int[m + 1][n + 1];
        int sum = 0;
        for (int i = 0; i < m; i++) sum += s1.charAt(i);
        for (int i = 0; i < n; i++) sum += s2.charAt(i);

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + (int) s1.charAt(i - 1);
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        return sum - dp[m][n] * 2;
    }

    //[739].每日温度
    public int[] dailyTemperatures(int[] temperatures) {
        int[] res = new int[temperatures.length];
        Stack<Integer> stack = new Stack<>();
        //单调递增减栈，遇到小的压掉它
        for (int i = temperatures.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                stack.pop();
            }

            res[i] = stack.isEmpty() ? 0 : stack.peek() - i;
            stack.push(i);
        }
        return res;
    }

    //[743].网络延迟时间
    public int networkDelayTime(int[][] times, int n, int k) {
        //最短路径
        int[] distTo = dijkstra(times, k, n);
        int res = 0;
        for (int i = 1; i <= n; i++) {
            if (distTo[i] == Integer.MAX_VALUE) {
                return -1;
            }
            res = Math.max(res, distTo[i]);
        }
        return res;
    }

    //dijkstra算法
    private int[] dijkstra(int[][] times, int start, int n) {
        List<int[]>[] graph = new LinkedList[n + 1];
        for (int i = 1; i < n + 1; i++) {
            graph[i] = new LinkedList<>();
        }
        for (int[] time : times) {
            graph[time[0]].add(new int[]{time[1], time[2]});
        }

        int[] distTo = new int[n + 1];
        Arrays.fill(distTo, Integer.MAX_VALUE);
        //id + 目前的最短距离
        Queue<int[]> queue = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
        queue.offer(new int[]{start, 0});
        distTo[start] = 0;
        while (!queue.isEmpty()) {
            int[] curNode = queue.poll();
            int id = curNode[0];
            int dist = curNode[1];
            if (dist > distTo[id]) {
                continue;
            }
            for (int[] next : graph[start]) {
                int nextId = next[1];
                int weight = next[2];
                int distToNext = distTo[id] + weight;
                if (distToNext < distTo[nextId]) {
                    distTo[nextId] = distToNext;
                    queue.offer(new int[]{nextId, distToNext});
                }
            }
        }
        return distTo;
    }

    //[747].至少是其他数字两倍的最大数
    public int dominantIndex(int[] nums) {
        if (nums.length == 1) return 0;
        int maxIndex = 0, secondIndex = -1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > nums[maxIndex]) {
                secondIndex = maxIndex;
                maxIndex = i;
            } else if (secondIndex == -1 || nums[i] > nums[secondIndex]) {
                secondIndex = i;
            }
        }
        return nums[maxIndex] >= 2 * nums[secondIndex] ? maxIndex : -1;
    }

    //[748].最短补全词
    public String shortestCompletingWord(String licensePlate, String[] words) {
        int[] need = new int[26];
        for (int i = 0; i < licensePlate.length(); i++) {
            char ch = licensePlate.charAt(i);
            if (Character.isLetter(ch)) {
                need[Character.toLowerCase(ch) - 'a']++;
            }
        }

        int index = -1;
        for (int i = 0; i < words.length; i++) {
            int[] cnt = new int[26];
            String word = words[i];
            for (int j = 0; j < word.length(); j++) {
                char ch = word.charAt(j);
                cnt[ch - 'a']++;
            }
            boolean valid = true;
            for (int j = 0; j < 26; j++) {
                if (cnt[j] < need[j]) {
                    valid = false;
                    break;
                }
            }
            if (valid && (index == -1 || words[i].length() < words[index].length())) {
                index = i;
            }
        }
        return words[index];
    }

    //[794].有效的井字游戏
    public boolean validTicTacToe(String[] board) {
        int xCount = 0, oCount = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (board[i].charAt(j) == 'X') {
                    xCount++;
                } else if (board[i].charAt(j) == 'O') {
                    oCount++;
                }
            }
        }
        if (xCount < oCount || xCount - oCount > 1) {
            return false;
        }
        if (win(board, 'X') && xCount - oCount != 1) {
            return false;
        }
        if (win(board, 'O') && xCount != oCount) {
            return false;
        }
        return true;
    }

    private boolean win(String[] board, char ch) {
        for (int i = 0; i < 3; i++) {
            if (board[i].charAt(0) == board[i].charAt(1) && board[i].charAt(1) == board[i].charAt(2) && board[i].charAt(0) == ch)
                return true;
            if (board[0].charAt(i) == board[1].charAt(i) && board[1].charAt(i) == board[2].charAt(i) && board[0].charAt(i) == ch)
                return true;
        }

        if (board[0].charAt(0) == board[1].charAt(1) && board[0].charAt(0) == board[2].charAt(2) && board[2].charAt(2) == ch)
            return true;
        if (board[0].charAt(2) == board[1].charAt(1) && board[0].charAt(2) == board[2].charAt(0) && board[2].charAt(0) == ch)
            return true;
        return false;
    }

    //[802].找到最终的安全状态
    public List<Integer> eventualSafeNodes(int[][] graph) {
        //存反图之后，那么入度为0的节点就是安全节点，而且不用顺序遍历
        //以出度为0的节点优先加入队列中，队列中的节点的邻居节点扣减出度，如果为0，加入队列。当队列为空，剩下的节点都是不合法节点。
        int n = graph.length;
        int[] outDegree = new int[n];
        List<Integer>[] inDegreeMap = new List[n];
        for (int i = 0; i < n; i++) {
            inDegreeMap[i] = new LinkedList<>();
        }
        for (int i = 0; i < n; i++) {
            int[] neighbors = graph[i];
            for (int neighbor : neighbors) {
                inDegreeMap[neighbor].add(i);
                outDegree[i]++;
            }
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (outDegree[i] == 0) {
                queue.offer(i);
            }
        }
        List<Integer> ans = new ArrayList<>();
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            ans.add(cur);
            //访问邻居节点，减出度
            for (int neighbor : inDegreeMap[cur]) {
                if (--outDegree[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }
        Collections.sort(ans);
        return ans;
    }

    //[807].保持城市天际线
    public int maxIncreaseKeepingSkyline(int[][] grid) {
        // 0 1 1
        // 2 1 1
        int n = grid.length;
        int[] left = new int[n];
        int[] upper = new int[n];
        for (int i = 0; i < n; i++) {
            upper[i] = Integer.MIN_VALUE;
            left[i] = Integer.MIN_VALUE;
            for (int j = 0; j < n; j++) {
                left[i] = Math.max(left[i], grid[i][j]);
                upper[i] = Math.max(upper[i], grid[j][i]);
            }
        }

        int res = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                res += Math.min(upper[j], left[i]) - grid[i][j];
            }
        }
        return res;
    }

    //[825].适龄的朋友
    public int numFriendRequests(int[] ages) {
        Arrays.sort(ages);
        int n = ages.length, ans = 0;
        for (int k = 0, i = 0, j = 0; k < n; k++) {
            //20,30,100,110,120
            //计算要么是全进来的，要么是全出去的，即计算进来，又计算出去，肯定有重复计算的数量
            //左边能到k的关系，i k
            while (i < k && !firstSelected(ages[i], ages[k])) i++;
            if (j < k) j = k;
            //右边能到k的关系，k j
            while (j < n && firstSelected(ages[j], ages[k])) j++;
            if (i < j) {
                //相同的情况多扣1
                ans += j - i - 1;
            }
        }
        return ans;
    }

    private boolean firstSelected(int x, int y) {
        if (y <= x * 0.5 + 7) return false;
        if (y > x) return false;
        if (y > 100 && x < 100) return false;
        return true;
    }

    //[846].一手顺子
    public static boolean isNStraightHand(int[] hand, int groupSize) {
        int n = hand.length;
        if (n < groupSize) return false;
        Arrays.sort(hand);
        Map<Integer, Integer> count = new HashMap<>();
        for (int poker : hand) {
            count.put(poker, count.getOrDefault(poker, 0) + 1);
        }

        //从小到大，选择最小的牌
        for (int start : hand) {
            //后面的牌被用掉了，可以继续找次小的牌
            if (count.get(start) == 0) continue;
            //连续选择后面牌，如果数量为0，则断牌了。
            for (int i = 0; i < groupSize; i++) {
                int target = start + i;
                if (count.getOrDefault(target, 0) == 0) {
                    return false;
                }
                count.put(target, count.get(target) - 1);
            }
        }
        return true;
    }

    //[851].喧闹和富有
    public int[] loudAndRich(int[][] richer, int[] quiet) {
        int n = quiet.length;
        //有钱->没钱 的邻接表， 从入度为0开始入队(从最有钱的开始遍历)，只有当前节点的比邻接点更安静，则更新邻接点的位置
        int[] ans = new int[n];
        List<Integer>[] graph = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            graph[i] = new ArrayList<>();
        }
        int[] indegree = new int[n];
        for (int[] rich : richer) {
            graph[rich[0]].add(rich[1]);
            indegree[rich[1]]++;
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            //初始值为本身
            ans[i] = i;
            //入度为0，都是些最有钱的人
            if (indegree[i] == 0) {
                queue.offer(i);
            }
        }
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            for (int next : graph[cur]) {
                //当前人找到的那个最安静的人 < 邻接人找到的那个最安静的人的时候，则更新。
                if (quiet[ans[cur]] < quiet[ans[next]]) {
                    ans[next] = ans[cur];
                }
                //加不进去没关系，但是度得维护下
                if (--indegree[next] == 0) {
                    queue.offer(next);
                }
            }
        }
        return ans;
    }

    //[875].爱吃香蕉的珂珂
    public int minEatingSpeed(int[] piles, int h) {
        int left = 1, right = Arrays.stream(piles).max().getAsInt();
        //首先肯定是左侧边界
        //hours(x) 是单调递减函数，不能死背模板
        // \
        //  \___   h
        //      \
        //        x (横轴)
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (hours(piles, mid) == h) {
                right = mid;
            } else if (hours(piles, mid) > h) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    private int hours(int[] piles, int speed) {
        int res = 0;
        for (int pile : piles) {
            if (pile % speed == 0) {
                res += pile / speed;
            } else {
                res += pile / speed + 1;
            }
        }
        return res;
    }

    //[876].链表的中间结点
    public ListNode middleNode(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    //[911].在线选举
    public static class TopVotedCandidate {
        int[] successor;
        int[] times;

        public TopVotedCandidate(int[] persons, int[] times) {
            int n = persons.length;
            successor = new int[n];
            Map<Integer, Integer> voteCounts = new HashMap<>();
            int topP = -1;
            for (int i = 0; i < n; i++) {
                int p = persons[i];

                voteCounts.put(p, voteCounts.getOrDefault(p, 0) + 1);
                //相等，也需要更新最新的领先者
                if (!voteCounts.containsKey(topP) || voteCounts.get(p) >= voteCounts.get(topP)) {
                    topP = p;
                }
                successor[i] = topP;
            }
            this.times = times;
        }

        public int q(int t) {
            //找到<=t 的时间
            int left = 0, right = times.length - 1;
            while (left < right) {
                int mid = left + (right - left + 1) / 2;
                if (times[mid] <= t) {
                    left = mid;
                } else {
                    right = mid - 1;
                }
            }
            return successor[left];
        }
    }

    //[913].猫和老鼠
    public int catMouseGame(int[][] graph) {
        int n = graph.length;
        int[][][] memo = new int[n][n][2 * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Arrays.fill(memo[i][j], -1);
            }
        }

        //dfs(i, j, k), i为鼠移动，j为猫移动，k为第几轮。 鼠刚开始在1，猫刚开始在2，k为0
        return dfs(memo, 1, 2, 0, graph);
    }

    private int dfs(int[][][] memo, int mouse, int cat, int turns, int[][] graph) {
        if (turns >= 2 * graph.length) return 0;
        if (memo[mouse][cat][turns] != -1) return memo[mouse][cat][turns];
        if (mouse == 0) return memo[mouse][cat][turns] = 1;
        if (mouse == cat) return memo[mouse][cat][turns] = 2;
        //鼠先动
        if (turns % 2 == 0) {
            //最坏的情况猫赢
            int ans = 2;
            for (int nexPos : graph[mouse]) {
                int ansNext = dfs(memo, nexPos, cat, turns + 1, graph);
                //优先取老鼠赢，再不济选平局，最后选择猫赢
                if (ansNext == 1) {
                    return memo[mouse][cat][turns] = 1;
                } else if (ansNext == 0) {
                    //平局，不返回，看最后结果
                    ans = 0;
                }
            }
            return memo[mouse][cat][turns] = ans;
        } else {
            //最坏的情况鼠赢
            int ans = 1;
            for (int nexPos : graph[cat]) {
                //猫要优先选择的位置不能为洞，不然老鼠就赢了
                if (nexPos != 0) {
                    int ansNext = dfs(memo, mouse, nexPos, turns + 1, graph);
                    //猫赢了，直接返回
                    if (ansNext == 2) {
                        return memo[mouse][cat][turns] = 2;
                    } else if (ansNext == 0) {
                        //平局，得看看其他
                        ans = 0;
                    }
                }
            }
            return memo[mouse][cat][turns] = ans;
        }
    }

    //[915].分割数组
    public int partitionDisjoint(int[] nums) {
        int n = nums.length;
        //左边的最大值 <= 右边的最小值
        int[] leftMax = new int[n];
        int[] rightMin = new int[n];
        int leftM = nums[0], rightM = nums[n - 1];
        for (int i = 0; i < n; i++) {
            if (nums[i] >= leftM) {
                leftM = nums[i];
            }
            leftMax[i] = leftM;
        }
        for (int i = n - 1; i >= 0; i--) {
            if (nums[i] <= rightM) {
                rightM = nums[i];
            }
            rightMin[i] = rightM;
        }

        for (int i = 0; i < n - 1; i++) {
            //左边的比右边的大的时候，可以继续扩，知道找到边界， 左边最大<=右边最小为止
            if (leftMax[i] <= rightMin[i + 1]) {
                return i + 1;
            }
        }
        return -1;
    }

    //[986].区间列表的交集
    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
        List<int[]> res = new ArrayList<>();
        int i = 0, j = 0;
        while (i < firstList.length && j < secondList.length) {
            int[] first = firstList[i];
            int[] second = secondList[j];
            //相交
            if (first[1] >= second[0] && second[1] >= first[0]) {
                res.add(new int[]{Math.max(first[0], second[0]), Math.min(first[1], second[1])});
            }

            if (first[1] > second[1]) {
                j++;
            } else {
                i++;
            }
        }
        return res.toArray(new int[][]{});
    }

    //[997].找到小镇的法官
    public int findJudge(int n, int[][] trust) {
        int[] outDegree = new int[n + 1];
        int[] inDegree = new int[n + 1];
        for (int[] item : trust) {
            outDegree[item[0]]++;
            inDegree[item[1]]++;
        }
        for (int i = 1; i <= n; i++) {
            if (inDegree[i] == n - 1 && outDegree[i] == 0) {
                return i;
            }
        }
        return -1;
    }

    //[1135].最低成本联通所有城市
    public int minimumCost(int N, int[][] connections) {
        UnionFind uf = new UnionFind(N + 1);
        //最低成本升序
        Arrays.sort(connections, (a, b) -> a[2] - b[2]);
        int res = Integer.MAX_VALUE;
        for (int[] connect : connections) {
            if (uf.connect(connect[0], connect[1])) {
                continue;
            }
            uf.union(connect[0], connect[1]);
            res += connect[2];
        }
        return uf.count == 2 ? res : -1;
    }

    //[1143].最长公共子序列
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        //长度为i的text1和长度为j的text2的最长公共子序列
        //字符串dp题目，以长度作为dp定义比较合适
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                //这边是结尾字符的索引
                if (text1.charAt(i - 1) == text2.charAt(i - 1)) {
                    //这边的是长度
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }

    //[1584].连接所有点的最小费用
    public int minCostConnectPoints(int[][] points) {
        int size = points.length;
        //点要转化为边集合
        List<int[]> edges = new ArrayList<>();
        for (int i = 0; i < size - 1; i++) {
            for (int j = i + 1; j < size; j++) {
                edges.add(new int[]{i, j, Math.abs(points[i][0] - points[j][0]) + Math.abs(points[i][1] - points[j][1])});
            }
        }

        Collections.sort(edges, Comparator.comparingInt(a -> ((int[]) a)[2]));

        int res = 0;
        UnionFind uf = new UnionFind(size);
        for (int[] edge : edges) {
            if (uf.connect(edge[0], edge[1])) {
                continue;
            }
            res += edge[2];

            uf.union(edge[0], edge[1]);
        }
        return res;
    }

    public static class UnionFind {
        private int[] parent;
        private int[] size;
        private int count;

        public UnionFind(int count) {
            parent = new int[count];
            size = new int[count];
            this.count = count;
            for (int i = 0; i < count; i++) {
                parent[i] = i;
                size[i] = 1;
            }
        }

        public int find(int x) {
            while (x != parent[x]) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }

        public boolean connect(int p, int q) {
            return find(p) == find(q);
        }

        public void union(int p, int q) {
            int rootP = find(p);
            int rootQ = find(q);
            if (rootP == rootQ) {
                return;
            }
            if (size[rootP] > size[rootQ]) {
                parent[rootQ] = rootP;
                size[rootP] += size[rootQ];
            } else {
                parent[rootP] = rootQ;
                size[rootQ] += size[rootP];
            }
            count--;
        }
    }

    //[1254].统计封闭岛屿的数目
    public int closedIsland(int[][] grid) {
        //0是陆地，1是水
        int m = grid.length, n = grid[0].length;
        //先淹掉周边的陆地，剩下的都是被水包围的
        for (int i = 0; i < m; i++) {
            dfsForClosedIsland(grid, i, 0);
            dfsForClosedIsland(grid, i, n - 1);
        }
        for (int j = 0; j < n; j++) {
            dfsForClosedIsland(grid, 0, j);
            dfsForClosedIsland(grid, m - 1, j);
        }

        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    res++;
                    dfsForClosedIsland(grid, i, j);
                }
            }
        }
        return res;
    }

    private void dfsForClosedIsland(int[][] grid, int x, int y) {
        int m = grid.length, n = grid[0].length;
        if (x < 0 || y < 0 || x >= m || y >= n) {
            return;
        }
        if (grid[x][y] == 1) {
            return;
        }
        //淹掉它
        grid[x][y] = 1;

        dfsForClosedIsland(grid, x - 1, y);
        dfsForClosedIsland(grid, x + 1, y);
        dfsForClosedIsland(grid, x, y - 1);
        dfsForClosedIsland(grid, x, y + 1);
    }

    //[1905].统计子岛屿
    public int countSubIslands(int[][] grid1, int[][] grid2) {
        int m = grid1.length, n = grid1[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                //前面是水，后面是陆，淹掉它
                if (grid1[i][j] == 0 && grid2[i][j] == 1) {
                    dfsForCountSubIslands(grid2, i, j);
                }
            }
        }
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                //剩下的都应该是子集
                if (grid2[i][j] == 1) {
                    res++;
                    dfsForCountSubIslands(grid2, i, j);
                }
            }
        }
        return res;
    }

    private void dfsForCountSubIslands(int[][] grid, int x, int y) {
        int m = grid.length, n = grid[0].length;
        if (x < 0 || x > m - 1 || y < 0 || y > n - 1) {
            return;
        }

        if (grid[x][y] == 0) {
            return;
        }
        grid[x][y] = 0;

        dfsForCountSubIslands(grid, x - 1, y);
        dfsForCountSubIslands(grid, x + 1, y);
        dfsForCountSubIslands(grid, x, y - 1);
        dfsForCountSubIslands(grid, x, y + 1);
    }

    //[1034].边界着色
    public int[][] colorBorder(int[][] grid, int row, int col, int color) {
        //原始颜色是跟color可能不相同的
        //遇到的边界其实是只有看到下一个是颜色不同的时候，才能确定的
        //岛屿可以任意修改，要么0，要么1，可以通过修改达到访问的目的
        //只更改边界颜色，意味着访问了之后不能更改颜色，不然回溯之后条件变化会导致判断条件失败，所以保存起来，之后再修改。
        int m = grid.length, n = grid[0].length;
        LinkedList<int[]> border = new LinkedList<>();
        dfsForColorBorder(grid, row, col, grid[row][col], border, new boolean[m][n]);

        for (int[] need : border) {
            grid[need[0]][need[1]] = color;
        }
        return grid;
    }

    private void dfsForColorBorder(int[][] grid, int x, int y, int originalColor, LinkedList<int[]> border, boolean[][] visit) {
        int m = grid.length, n = grid[0].length;
        int[][] directions = new int[][]{{0, -1}, {0, 1}, {-1, 0}, {1, 0}};

        visit[x][y] = true;
        boolean isBorder = false;
        for (int[] direct : directions) {
            int nx = x + direct[0];
            int ny = y + direct[1];

            if (!(nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == originalColor)) {
                isBorder = true;
            } else if (!visit[nx][ny]) {
                dfsForColorBorder(grid, nx, ny, originalColor, border, visit);
            }
        }
        if (isBorder) {
            border.addLast(new int[]{x, y});
        }
    }

    private int[][] colorBorderV2(int[][] grid, int row, int col, int color) {
        int m = grid.length, n = grid[0].length;
        dfsForColorBorder(grid, row, col, color, grid[row][col], new boolean[m][n]);
        return grid;
    }

    private void dfsForColorBorder(int[][] grid, int x, int y, int color, int originalColor, boolean[][] visit) {
        int m = grid.length, n = grid[0].length;
        int[][] directions = new int[][]{{0, -1}, {0, 1}, {-1, 0}, {1, 0}};

        visit[x][y] = true;
        for (int[] direct : directions) {
            int nx = x + direct[0];
            int ny = y + direct[1];
            //访问过的，可能处理过边界，所以不访问了
            if (nx >= 0 && ny >= 0 && nx < m && ny < n && !visit[nx][ny] && grid[nx][ny] == originalColor) {
                //找到跟原始相同的颜色，不是边界，则继续递归
                dfsForColorBorder(grid, nx, ny, color, originalColor, visit);
            } else if (nx >= 0 && ny >= 0 && nx < m && ny < n && !visit[nx][ny] && grid[nx][ny] != originalColor || nx < 0 || ny < 0 || nx >= m || ny >= n) {
                //找到跟原始不相同的颜色，或者超出边界，直接修改颜色
                grid[x][y] = color;
            }
        }
    }

    //[2074].反转偶数长度组的节点
    public ListNode reverseEvenLengthGroups(ListNode head) {
        int count = 0;
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode pre = dummy;
        ListNode cur = head;
        while (cur != null) {
            count++;

            int length = 0;
            ListNode next = cur;
            while (length < count && next != null) {
                next = next.next;
                length++;
            }
            if (length % 2 == 0) {
                //cur就是要转化的头节点, cur一直在当前节点上没变
                for (int i = 0; i < length - 1; i++) {
                    ListNode removed = cur.next;
                    cur.next = cur.next.next;
                    removed.next = pre.next;
                    pre.next = removed;
                }
                pre = cur;
                cur = cur.next;
            } else {
                for (int i = 0; i < length; i++) {
                    cur = cur.next;
                    pre = pre.next;
                }
            }
        }
        return dummy.next;
    }

    //[1816].截断句子
    public String truncateSentence(String s, int k) {
        int count = 0, end = 0;
        for (int i = 1; i <= s.length(); i++) {
            if (i == s.length() || s.charAt(i) == ' ') {
                count++;
                if (count == k) {
                    end = i;
                    break;
                }
            }
        }
        return s.substring(0, end);
    }

    //[1288].删除被覆盖的区间
    public int removeCoveredIntervals(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> a[0] == b[0] ? b[1] - a[1] : a[0] - b[0]);
        //只需要一个右侧边界是因为已经按照左边界排序了。
        int right = intervals[0][1];
        int res = 0;
        for (int i = 1; i < intervals.length; i++) {
            //此时右边界包含，必定，左边界也被包含
            if (right >= intervals[i][1]) {
                res++;
            } else {
                right = intervals[i][1];
            }
        }
        return intervals.length - res;
    }

    //[1011].在 D 天内送达包裹的能力
    public int shipWithinDays(int[] weights, int days) {
        int left = Arrays.stream(weights).max().getAsInt(), right = Arrays.stream(weights).sum();
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (days(weights, mid) == days) {
                right = mid;
            } else if (days(weights, mid) > days) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    private int days(int[] weights, int capacity) {
        int res = 0;
        for (int i = 0; i < weights.length; ) {
            int cap = capacity;
            //不断的尝试放到容量中，不够就跳出，增加一天
            while (i < weights.length) {
                if (cap < weights[i]) {
                    break;
                } else {
                    cap -= weights[i];
                }
                i++;
            }
            res++;
        }
        return res;
    }

    //[2073].买票需要的时间
    public int timeRequiredToBuy(int[] tickets, int k) {
        int need = tickets[k];
        int res = 0;
        //2 3 4 1, 如果k=1, i = 2因为在它后面，最多也就能购2票，k就结束了
        for (int i = 0; i < tickets.length; i++) {
            if (i <= k) {
                res += Math.min(need, tickets[i]);
            } else {
                res += Math.min(need - 1, tickets[i]);
            }
        }
        return res;
    }

    //[面试题 17.12].BiNode
    private TreeNode pre = null, head = null;

    public TreeNode convertBiNode(TreeNode root) {
        //二叉搜索树，中序遍历一定是从小到大的
        //应该先找到前一个节点，然后把当前节点作为它的右孩子，最后左孩子置空
        if (root == null) return null;
        convertBiNode(root.left);
        if (pre == null) {
            head = root;
        } else {
            pre.right = root;
        }
        pre = root;
        root.left = null;
        convertBiNode(root.right);
        return head;
    }

    //[1610].可见点的最大数目
    public int visiblePoints(List<List<Integer>> points, int angle, List<Integer> location) {
        //首先转化极坐标，其次扩大坐标范围，滑动窗口更新可见数量
        List<Double> polarDegrees = new ArrayList<>();
        int x = location.get(0), y = location.get(1), sameCount = 0;
        for (List<Integer> point : points) {
            if (x == point.get(0) && y == point.get(1)) {
                sameCount++;
                continue;
            }
            //是弧度值，值范围(-PI,PI]
            polarDegrees.add(Math.atan2(point.get(1) - y, point.get(0) - x));
        }
        int n = polarDegrees.size();
        //只从小到大排序0到n-1
        Collections.sort(polarDegrees);
        //当有-PI的时候，与PI是重叠的，解决第三象限和第二象限不能连续扫描的问题，加上2PI，就是-PI变成了PI值，又可以和第二象限连续了。
        for (int i = 0; i < n; i++) {
            polarDegrees.add(polarDegrees.get(i) + 2 * Math.PI);
        }

        int left = 0, right = 0, maxCount = 0;
        //换算成弧度值
        double range = angle * Math.PI / 180;
        //滑动窗口遍历数组，不断扩大右边区域，发现不在范围内，缩小左边区域，区间值就是点的个数
        while (right < 2 * n) {
            while (left < right && polarDegrees.get(right) - polarDegrees.get(left) > range) {
                left++;
            }
            //更新值
            maxCount = Math.max(maxCount, right - left + 1);
            right++;
        }

        return maxCount + sameCount;
    }

    //[1518].换酒问题
    public int numWaterBottles(int numBottles, int numExchange) {
        int bottles = numBottles, left = numBottles;
        while (left >= numExchange) {
            //可以换多少瓶新的
            int empty = left / numExchange;
            bottles += empty;
            //剩余的变成了新换的+ 剩下换不了的
            left = empty + left % numExchange;
        }
        return bottles;
    }

    //[1109].航班预订统计
    public int[] corpFlightBookings(int[][] bookings, int n) {
        Difference difference = new Difference(new int[n]);
        for (int[] booking : bookings) {
            difference.insert(booking[0] - 1, booking[1] - 1, booking[2]);
        }
        return difference.result();
    }

    //[1094].拼车
    public boolean carPooling(int[][] trips, int capacity) {
        //可以从1开始，0位置没有人乘车也没关系，没人<容量
        Difference difference = new Difference(new int[1001]);
        for (int[] trip : trips) {
            //[1, 5] 5下车，说明区间在1,4之间，5重新上车乘客数才是要跟容量比
            difference.insert(trip[1], trip[2] - 1, trip[0]);
        }
        int[] result = difference.result();
        for (int each : result) {
            if (each > capacity) {
                return false;
            }
        }
        return true;
    }

    //[1375].二进制字符串前缀一致的次数
    public int numTimesAllBlue(int[] flips) {
        int maxPos = 0, cnt = 0;
        //[3,2,4,1,5] 最大的位置等于当前位置就意味着是一个前缀字符串
        for (int i = 0; i < flips.length; i++) {
            maxPos = Math.max(maxPos, flips[i]);
            if (i + 1 == maxPos) {
                cnt++;
            }
        }
        return cnt;
    }

    //[2075].解码斜向换位密码
    public String decodeCiphertext(String encodedText, int rows) {
        if (encodedText.isEmpty() || rows == 1) return encodedText;
        int cols = encodedText.length() / rows;
        int i = 0, j = 0, count = 0;
        StringBuilder sb = new StringBuilder();
        while (j < cols) {
            sb.append(encodedText.charAt(i * cols + j));
            count++;
            i++;
            j++;
            //右下角一个元素标记着最多还有一斜行，最多cols个元素，达不到最后列，就可以结束了。
            if (count % rows == 0) {
                i = 0;
                j = count / rows;
            }
        }
        //只能剔除掉最后的空格
        j = sb.length() - 1;
        while (sb.charAt(j) == ' ') {
            j--;
        }
        return sb.delete(j + 1, sb.length()).toString();
    }

    //[5956].找出数组中的第一个回文字符串
    public String firstPalindrome(String[] words) {
        for (int i = 0; i < words.length; i++) {
            if (isPalindrome(words[i])) {
                return words[i];
            }
        }
        return "";
    }

    private boolean isPalindrome(String word) {
        int left = 0, right = word.length() - 1;
        while (left < right) {
            if (word.charAt(left) == word.charAt(right)) {
                left++;
                right--;
            } else {
                return false;
            }
        }
        return true;
    }

    //[5957].向字符串添加空格
    public String addSpaces(String s, int[] spaces) {
        StringBuilder sb = new StringBuilder(s);
        int count = 0;
        for (int space : spaces) {
            sb.insert(space + count, ' ');
            count++;
        }
        return sb.toString();
    }

    //[5958].股票平滑下跌阶段的数目
    public long getDescentPeriods(int[] prices) {
        int n = prices.length;
        //从第1个开始，少算了一个数
        //count前面子集的个数，默认从1开始，本身也是递减的
        long ans = 1, count = 1;
        for (int i = 1; i < n; i++) {
            if (prices[i - 1] == prices[i] + 1) {
                count++;
            } else {
                count = 1;
            }
            ans += count;
        }

        return ans;
    }

    //[5959].使数组 K 递增的最少操作次数
    public int kIncreasing(int[] arr, int k) {
        int n = arr.length, ans = 0;
        for (int i = 0; i < k; i++) {
            List<Integer> list = new ArrayList<>();
            for (int j = i; j < n; j += k) {
                list.add(arr[j]);
            }
            //如果是一个，则不需要计数
            if (list.size() == 1) continue;

            //最大递增子序列
            int[] dp = new int[list.size()];
            //最大默认有0堆
            int maxLen = 0;
            for (int num : list) {
                //右侧边界，大于的模版
                int left = 0, right = maxLen;
                while (left < right) {
                    int mid = left + (right - left) / 2;
                    if (dp[mid] <= num) {
                        left = mid + 1;
                    } else {
                        right = mid;
                    }
                }
                //放最顶上去
                dp[left] = num;
                //需要新建一个堆
                if (maxLen == left) {
                    maxLen++;
                }
            }
            ans += list.size() - maxLen;
        }
        return ans;
    }

    //[1154].一年中的第几天
    public static int dayOfYear(String date) {
        String[] vars = date.split("-");
        int year = Integer.parseInt(vars[0]);
        int month = Integer.parseInt(vars[1]);
        int day = Integer.parseInt(vars[2]);
        int[] days = new int[]{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        if (year % 400 == 0 || (year % 4 == 0 && year % 100 != 0)) {
            days[1] += 1;
        }
        int ans = 0;
        for (int i = 0; i < month - 1; i++) {
            ans += days[i];
        }
        return ans + day;
    }

    //Morris遍历
    private void preOrderMorris(TreeNode root) {
        TreeNode cur = root, rightMost = null;
        while (cur != null) {
            if (cur.left == null) {
                //是从这边直接遍历到原来的根节点的
                cur = cur.right;
            } else {
                rightMost = cur.left;
                while (rightMost.right != null && rightMost.right != cur) {
                    rightMost = rightMost.right;
                }
                //第一次访问，创建到根节点的快捷路径
                if (rightMost.right == null) {
                    rightMost.right = cur;
                    cur = cur.left;
                } else {
                    //第二次了，直接断开
                    rightMost.right = null;
                    cur = cur.right;
                }
            }
        }
    }

    //[1044].最长重复子串
    public static String longestDupSubstring(String s) {
        int n = s.length(), P = 1313131;
        long[] p = new long[n + 1], h = new long[n + 1];
        p[0] = 1;
        for (int i = 0; i < n; i++) {
            //P次方数组，P的i次方
            p[i + 1] = p[i] * P;
            //s(0..i-1)的hash值
            h[i + 1] = h[i] * P + s.charAt(i);
        }
        String ans = "";
        int left = 1, right = n - 1;
        while (left < right) {
            int len = left + (right - left + 1) / 2;
            String res = check(s, len, p, h);
            if (res.length() == 0) right = len - 1;
            else left = len;
            ans = res.length() > ans.length() ? res : ans;
        }
        return ans;
    }

    private static String check(String s, int len, long[] p, long[] h) {
        Set<Long> set = new HashSet<>();
        for (int i = 1; i + len - 1 <= s.length(); i++) {
            int j = i + len - 1;
            //区间: i-1 ~ j-1
            //abcabc, 为n进制数，最右边的是a * n^0，所以h[i-1]要往左边推j-i+1位才可以。
            long hash = h[j] - h[i - 1] * p[j - i + 1];
            if (set.contains(hash)) {
                return s.substring(i - 1, j);
            } else {
                set.add(hash);
            }
        }
        return "";
    }

    //[1705].吃苹果的最大数目
    public int eatenApples(int[] apples, int[] days) {
        //apples = [1,2,3,5,2], days = [3,2,1,4,2] //输出：7
        //1. 可能没有苹果产生，2.苹果到了某一天有叠加，选快过期的是最好的选择
        //保质期 + 数量
        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> a[0] - b[0]);
        //当前时间没有苹果产生， 不加入队列； 当前有苹果产生，加入队列。
        //如果当前，有保质期内的苹果，那么就可以吃一个。
        int curTime = 0, n = apples.length, ans = 0;
        while (curTime < n || !queue.isEmpty()) {
            if (curTime < n && apples[curTime] > 0) {
                //保质期：当前时间 + 最长的时间 -1
                queue.offer(new int[]{curTime + days[curTime] - 1, apples[curTime]});
            }
            //保质期比现在早的，全部不要
            while (!queue.isEmpty() && curTime > queue.peek()[0]) {
                queue.poll();
            }
            if (!queue.isEmpty()) {
                int[] cur = queue.poll();
                //减了一个苹果之后，如果还有并且没有超过有效期，则重新添加进去
                if (--cur[1] > 0 && cur[0] > curTime) {
                    queue.offer(cur);
                }
                ans++;
            }
            curTime++;
        }
        return ans;
    }

    //[1609].奇偶树
    public boolean isEvenOddTree(TreeNode root) {
        //就是不用BFS, 在处理每一层的时候，需要获取到前面一个节点。
        return dfsForIsEvenOddTree(root, 0, new HashMap<>());
    }

    private boolean dfsForIsEvenOddTree(TreeNode root, int level, Map<Integer, Integer> levelMap) {
        boolean flag = level % 2 == 0;
        int prev = levelMap.getOrDefault(level, flag ? Integer.MIN_VALUE : Integer.MAX_VALUE);
        if (flag && (root.val % 2 == 0 || root.val <= prev)) return false;
        else if (!flag && (root.val % 2 != 0 || root.val >= prev)) return false;
        levelMap.put(level, root.val);
        if (root.left != null && !dfsForIsEvenOddTree(root.left, level + 1, levelMap)) return false;
        if (root.right != null && !dfsForIsEvenOddTree(root.right, level + 1, levelMap))
            return false;
        return true;
    }

    //[1169].查询无效交易
    public List<String> invalidTransactions(String[] transactions) {
        Set<Integer> deleteIndex = new HashSet<>();
        List<String[]> strList = new ArrayList<>();
        int n = transactions.length;

        for (int i = 0; i < n; i++) {
            strList.add(transactions[i].split(","));
        }

        for (int i = 0; i < n; i++) {
            String[] cur = strList.get(i);

            if (Integer.parseInt(cur[2]) > 1000) {
                deleteIndex.add(i);
            }

            for (int j = i + 1; j < n; j++) {
                String[] next = transactions[j].split(",");
                if (cur[0].equals(next[0])
                        && !cur[3].equals(next[3])
                        && Math.abs(Integer.parseInt(cur[1]) - Integer.parseInt(next[1])) <= 60) {
                    deleteIndex.add(i);
                    deleteIndex.add(j);
                }
            }
        }

        List<String> result = new ArrayList<>();
        for (Integer index : deleteIndex) {
            result.add(transactions[index]);
        }
        return result;
    }

    //[1876].长度为三且各字符不同的子字符串
    public int countGoodSubstrings(String s) {
        int n = s.length();
        if (n < 3) return 0;
        int ans = 0;
        StringBuilder window = new StringBuilder();
        for (int left = 0, right = 0; right < n; ) {
            window.append(s.charAt(right++));

            while (right - left >= 3) {
                if (window.charAt(0) != window.charAt(1)
                        && window.charAt(1) != window.charAt(2)
                        && window.charAt(0) != window.charAt(2)) {
                    ans++;
                }
                window.deleteCharAt(0);
                left++;
            }
        }
        return ans;
    }

    //[1078].Bigram 分词
    public String[] findOcurrences(String text, String first, String second) {
        String[] words = text.split(" ");
        List<String> result = new ArrayList<>();
        for (int i = 2; i < words.length; i++) {
            if (words[i - 2].equals(first) && words[i - 1].equals(second)) {
                result.add(words[i]);
            }
        }
        return result.toArray(new String[]{});
    }

    //[5963].反转两次的数字
    public boolean isSameAfterReversals(int num) {
        return num > 0 ? num % 10 != 0 : true;
    }

    //[5964].执行所有后缀指令
    public int[] executeInstructions(int n, int[] startPos, String s) {
        int[] res = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            res[i] = dfsForExecuteInstructions(s, i, startPos[0], startPos[1], n);
        }
        return res;
    }

    private int dfsForExecuteInstructions(String s, int start, int x, int y, int n) {
        if (start >= s.length()) return 0;
        char ch = s.charAt(start);
        int nextX = x, nextY = y;
        if (ch == 'U') {
            nextX = x - 1;
        } else if (ch == 'R') {
            nextY = y + 1;
        } else if (ch == 'L') {
            nextY = y - 1;
        } else {
            nextX = x + 1;
        }
        if (nextX < 0 || nextY < 0 || nextX >= n || nextY >= n) {
            return 0;
        }
        return dfsForExecuteInstructions(s, start + 1, nextX, nextY, n) + 1;
    }

    //[5965].相同元素的间隔之和
    public long[] getDistances(int[] arr) {
        //key,value: key代表元素值， value[0]元素值相等，前面最近的一次下标，value[1]元素值相等的有多少个
        Map<Integer, int[]> prefix = new HashMap<>();
        Map<Integer, int[]> suffix = new HashMap<>();
        int n = arr.length;
        //从前往后，元素相等的距离差之和 = （个数*最近一次的距离） + 最近一次的值
        long[] prefixDistanceSum = new long[n];
        for (int i = 0; i < n; i++) {
            int num = arr[i];
            int[] value = prefix.getOrDefault(num, new int[2]);
            int perIndex = value[0];
            int count = value[1];
            if (count != 0) {
                prefixDistanceSum[i] = prefixDistanceSum[perIndex] + count * (i - perIndex);
            }
            value[0] = i;
            value[1]++;
            prefix.put(num, value);
        }
        //从后往前，元素相等的距离差之和 = （个数*最近一次的距离） + 最近一次的值
        long[] suffixDistanceSum = new long[n];
        for (int i = n - 1; i >= 0; i--) {
            int num = arr[i];
            int[] value = suffix.getOrDefault(num, new int[2]);
            int perIndex = value[0];
            int count = value[1];
            if (count != 0) {
                suffixDistanceSum[i] = suffixDistanceSum[perIndex] + count * (perIndex - i);
            }
            value[0] = i;
            value[1]++;
            suffix.put(num, value);
        }
        //最终的距离等于从前往后 + 从后往前
        long[] res = new long[n];
        for (int i = 0; i < n; i++) {
            res[i] = prefixDistanceSum[i] + suffixDistanceSum[i];
        }
        return res;
    }

    //[1995].统计特殊四元组
    public int countQuadruplets(int[] nums) {
        //a + b = d - c, 以b逆序遍历，a从0到b-1， d从b+1到n-1
        Map<Integer, Integer> countMap = new HashMap<>();
        int n = nums.length, ans = 0;
        for (int b = n - 3; b >= 1; b--) {
            for (int d = b + 2; d < n; d++) {
                int c = b + 1;
                int d_c = nums[d] - nums[c];
                countMap.put(d_c, countMap.getOrDefault(d_c, 0) + 1);
            }
            for (int a = 0; a < b; a++) {
                ans += countMap.getOrDefault(nums[a] + nums[b], 0);
            }
        }
        return ans;
    }

    //[1446].连续字符
    public int maxPower(String s) {
        if (s.length() == 0) return 0;
        int left = 0, right = 0;
        int res = 0;
//        while (right < s.length()) {
//            if (s.charAt(right) == s.charAt(left)) {
//                right++;
//                res = Math.max(res, right - left);
//            } else {
//                left = right;
//            }
//        }
//        return res;
        //上面的写法每次都得重新设置最大值
        while (left < s.length()) {
            right = left;
            while (right < s.length() && s.charAt(left) == s.charAt(right)) right++;
            res = Math.max(res, right - left);
            //换到不想等的位置继续
            left = right;
        }
        return res;
    }

    //[1005].K 次取反后最大化的数组和
    public int largestSumAfterKNegations(int[] nums, int k) {
        Arrays.sort(nums);
        int n = nums.length;
        int sum = 0;
        //-5 -4 -1 2 3 4    k = 5 k剩余为偶数
        //-5 -4 -1 2 3 4    k = 2  k没剩余
        for (int i = 0; i < n; i++) {
            if (nums[i] < 0 && k > 0) {
                nums[i] = -nums[i];
                k--;
            }
            sum += nums[i];
        }
        //k没有剩余
        if (k == 0) return sum;
            //k剩余偶数，可以全部抵扣掉
        else if (k % 2 == 0) return sum;
            //k剩余奇数，可以只变更最小的那个数
        else {
            Arrays.sort(nums);
            //原来是加的，现在要变成负值，需要扣原来加的。
            return sum - 2 * nums[0];
        }
    }

    //[2022].将一维数组转变成二维数组
    public int[][] construct2DArray(int[] original, int m, int n) {
        int len = original.length;
        if (len != m * n) return new int[0][0];
        int[][] res = new int[m][n];
        for (int i = 0; i < len; i++) {
            int x = i / n;
            int y = i % n;
            res[x][y] = original[i];
        }
        return res;
    }

    //[1036].逃离大迷宫
    public boolean isEscapePossible(int[][] blocked, int[] source, int[] target) {
        int n = blocked.length;
        Set<Long> vis = new HashSet<>();
        for (int[] p : blocked) {
            vis.add(p[0] * 131l + p[1]);
        }
        //最优的情况下，利用边界使得包围圈最大化，最多只有n-1 + n-2 + ... + 1。斜的n个不算，超过一个就是可以逃离。
        int MAX = (n - 1) * n / 2;
        return check(source, target, MAX, vis) && check(target, source, MAX, vis);
    }

    private boolean check(int[] source, int[] target, int n, Set<Long> vis) {
        Set<Long> set = new HashSet<>();
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(source);
        set.add(source[0] * 131l + source[1]);

        int[][] dir = new int[][]{{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
        while (!queue.isEmpty() && set.size() <= n) {
            int[] cur = queue.poll();
            if (cur[0] == target[0] && cur[1] == target[1]) return true;
            for (int[] d : dir) {
                int nextX = cur[0] + d[0];
                int nextY = cur[1] + d[1];
                if (nextX < 0 || nextY < 0 || nextX >= (int) 1e6 || nextY >= (int) 1e6) continue;
                long hash = nextX * 131l + nextY;
                if (vis.contains(hash)) continue;
                if (set.contains(hash)) continue;
                set.add(hash);
                queue.offer(new int[]{nextX, nextY});
            }
        }
        return set.size() > n;
    }

    //[1185].一周中的第几天
    public String dayOfTheWeek(int day, int month, int year) {
        String[] week = new String[]{"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"};
        int[] months = new int[]{-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        //年份带来的天数 + 闰年带来的差异，1972年是闰年
        int days = (year - 1971) * 365 + (year - 1969) / 4;
        for (int i = 1; i < month; i++) {
            days += months[i];
        }
        //当年是闰年，加一天
        if (month >= 3 && (year % 400 == 0 || (year % 4 == 0 && year % 100 != 0))) {
            days++;
        }
        days += day;
        //1970年12月31号，星期四，偏移4天，但是week是从0开始，实际偏移3
        return week[(days + 3) % 7];
    }

    //[1576].替换所有的问号
    public static String modifyString(String s) {
        int n = s.length();
        StringBuilder sb = new StringBuilder(s);
        for (int i = 0; i < n; i++) {
            //前后都不相同，需要第三次猜测数字
            for (int k = 0; k < 3 && sb.charAt(i) == '?'; k++) {
                boolean ok = true;
                char ch = (char) ('a' + k);
                //问号也是不同的
                if (i - 1 >= 0 && sb.charAt(i - 1) == ch) ok = false;
                if (i + 1 < n && sb.charAt(i + 1) == ch) ok = false;
                if (ok) sb.setCharAt(i, ch);
            }
        }
        return sb.toString();
    }

    //[1614].括号的最大嵌套深度
    public int maxDepth(String s) {
        int count = 0, res = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') count++;
            else if (s.charAt(i) == ')') count--;
            res = Math.max(count, res);
        }
        return res;
    }

    //[剑指 Offer 09].用两个栈实现队列
    public class CQueue {

        Stack<Integer> s1;
        Stack<Integer> s2;

        public CQueue() {
            s1 = new Stack<>();
            s2 = new Stack<>();
        }

        public void appendTail(int value) {
            s1.push(value);
        }

        //没有peek操作，所以应该减少push的操作次数
        public int deleteHead() {
            if (s2.isEmpty()) {
                while (!s1.isEmpty()) {
                    s2.push(s1.pop());
                }
            }
            if (s2.isEmpty()) {
                return -1;
            } else {
                return s2.pop();
            }
        }
    }

    //[155].最小栈
    //[剑指 Offer 30].包含min函数的栈
    public class MinStack {

        Stack<Long> diffStack;
        long min;

        /**
         * initialize your data structure here.
         */
        public MinStack() {
            diffStack = new Stack<>();
        }

        public void push(int x) {
            if (diffStack.isEmpty()) {
                diffStack.push(0l);
                min = x;
            } else {
                //当前值比最小值还要小，说明需要更新最小值
                if (x < min) {
                    //更新min值
                    diffStack.push(x - min);
                    min = x;
                } else {
                    diffStack.push(x - min);
                }
            }
        }

        public void pop() {
            if (!diffStack.isEmpty()) {
                //差异值为负值，说明最小值发生了变化，pop之后需要变更
                if (diffStack.peek() < 0) {
                    min = min - diffStack.peek();
                }
                diffStack.pop();
            }
        }

        public int top() {
            //差异值为负值，说明min就是当前值
            if (diffStack.peek() < 0) {
                return (int) min;
            } else {
                return (int) (min + diffStack.peek());
            }
        }

        public int min() {
            return (int) min;
        }
    }

    //[5976].检查是否每一行每一列都包含全部整数
    public boolean checkValid(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < n; i++) {
            int[] rows = new int[n + 1];
            int[] cols = new int[n + 1];
            for (int j = 0; j < n; j++) {
                int row = matrix[i][j];
                if (rows[row] >= 1) {
                    return false;
                }
                rows[row]++;

                int col = matrix[j][i];
                if (cols[col] >= 1) {
                    return false;
                }
                cols[col]++;
            }
        }
        return true;
    }

    //[5977].最少交换次数来组合所有的 1 II
    public int minSwaps(int[] nums) {
        int count = 0, n = nums.length;
        for (int d : nums) {
            if (d == 1) count++;
        }
        int left = 0, right = 0;
        int zeros = 0, min = count;
        //窗口内的0的个数即为交换次数，最小值就是最少交换次数
        while (right < 2 * n) {
            if (nums[right % n] == 0) {
                zeros++;
            }
            if (right - left + 1 >= count) {
                min = Math.min(min, zeros);
                //left如果是0，0计数减一
                if (nums[left % n] == 0) {
                    zeros--;
                }
                left++;
            }
            right++;
        }
        return min;
    }

    //[5978].统计追加字母可以获得的单词数
    public int wordCount(String[] startWords, String[] targetWords) {
        //题目说了，不会有重复字符，所以可以用int进行压缩
        int res = 0;
        Set<Integer> hash = new HashSet<>();
        for (String startWord : startWords) {
            int num = 0;
            for (char ch : startWord.toCharArray()) {
                num |= 1 << ch - 'a';
            }
            hash.add(num);
        }
        for (String targetWord : targetWords) {
            int num = 0;
            for (char ch : targetWord.toCharArray()) {
                //26进制去占位
                num |= 1 << ch - 'a';
            }
            for (char ch : targetWord.toCharArray()) {
                //移除每一位字符判断是否存在
                if (hash.contains(num ^ 1 << ch - 'a')) {
                    res++;
                    break;
                }
            }
        }
        return res;
    }

    //[1220].统计元音字母序列的数量
    public int countVowelPermutation(int n) {
        // 字符串中的每个字符都应当是小写元音字母（'a', 'e', 'i', 'o', 'u'）
        // 每个元音 'a' 后面都只能跟着 'e'
        // 每个元音 'e' 后面只能跟着 'a' 或者是 'i'
        // 每个元音 'i' 后面 不能 再跟着另一个 'i'
        // 每个元音 'o' 后面只能跟着 'i' 或者是 'u'
        // 每个元音 'u' 后面只能跟着 'a'
        int MOD = (int) 1e9 + 7;
        //以i开头的字符串，以元音字母结尾的字符数量
        long[][] dp = new long[n][5];
        Arrays.fill(dp[n - 1], 1);
        for (int i = n - 2; i >= 0; i--) {
            //以a开头
            dp[i][0] = dp[i + 1][1];
            //以e开头
            dp[i][1] = dp[i + 1][0] + dp[i + 1][2];
            //以i开头
            dp[i][2] = dp[i + 1][0] + dp[i + 1][1] + dp[i + 1][3] + dp[i + 1][4];
            //以o开头
            dp[i][3] = dp[i + 1][2] + dp[i + 1][4];
            //以u开头
            dp[i][4] = dp[i + 1][0];
            for (int j = 0; j < 5; j++) {
                dp[i][j] %= MOD;
            }
        }
        long ans = 0;
        for (int i = 0; i < 5; i++) {
            ans += dp[0][i];
        }
        return (int) (ans % MOD);
    }

    //[1220].统计元音字母序列的数量(矩阵快速幂)
    private int countVowelPermutation2(int n) {
        int MOD = (int) 1e9 + 7;
        long[][] mat = new long[][]{
                {0, 1, 0, 0, 0},
                {1, 0, 1, 0, 0},
                {1, 1, 0, 1, 1},
                {0, 0, 1, 0, 1},
                {1, 0, 0, 0, 0}};
        long[][] ans = new long[][]{{1}, {1}, {1}, {1}, {1}};

        //矩阵快速幂
        int x = n - 1;
        while (x != 0) {
            //注意是mat在前面
            if ((x & 1) != 0) ans = mul(mat, ans);
            mat = mul(mat, mat);
            x >>= 1;
        }

        long res = 0;
        for (int i = 0; i < 5; i++) {
            res += ans[i][0];
        }
        return (int) (res % MOD);
    }

    //矩阵快速幂
    private long[][] mul(long[][] a, long[][] b) {
        int MOD = (int) 1e9 + 7;
        int r = a.length, c = b[0].length, z = b.length;
        long[][] res = new long[r][c];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                for (int k = 0; k < z; k++) {
                    res[i][j] += a[i][k] * b[k][j];
                    res[i][j] %= MOD;
                }
            }
        }
        return res;
    }

    //[1629].按键持续时间最长的键
    public char slowestKey(int[] releaseTimes, String keysPressed) {
        int idx = 0, max = releaseTimes[0];
        for (int i = 1; i < keysPressed.length(); i++) {
            int r = releaseTimes[i] - releaseTimes[i - 1];
            if (r > max) {
                idx = i;
                max = r;
            } else if (r == max && keysPressed.charAt(i) > keysPressed.charAt(idx)) {
                idx = i;
            }
        }
        return keysPressed.charAt(idx);
    }

    //[1716].计算力扣银行的钱
    public int totalMoney(int n) {
        int weekNum = n / 7;
        int dayOfWeek = n % 7;
        //星期有多少天，每增加一周，增加7天，第三周为14天，等差数列求和
        int moneyOfWeeks = weekNum > 0 ? 28 * weekNum + 7 * weekNum * (weekNum - 1) / 2 : 0;
        //第几个星期，首项就是多少
        int moneyOfDays = dayOfWeek * (weekNum + 1) + dayOfWeek * (dayOfWeek - 1) / 2;
        return moneyOfDays + moneyOfWeeks;
    }

    //[5980].将字符串拆分为若干长度为 k 的组
    public static String[] divideString(String s, int k, char fill) {
        int g = s.length() / k;
        int n = (s.length() % k == 0) ? g : g + 1;
        String[] ans = new String[n];
        for (int i = 0; i < n; i++) {
            if (s.length() >= (k * (i + 1))) {
                ans[i] = s.substring(k * i, k * (i + 1));
            } else {
                int count = k * (i + 1) - s.length();
                StringBuilder sb = new StringBuilder(s.substring(k * i));
                while (count-- > 0) {
                    sb.append(fill);
                }
                ans[i] = sb.toString();
            }
        }
        return ans;
    }

    //[5194].得到目标值的最少行动次数
    public static int minMoves(int target, int maxDoubles) {
        int ans = 0;
        //当时脑子短路，居然用动态规划来做，优先想O(n)的办法哇
        //奇数一定是+1，偶数一定是优先通过倍数来搞
        while (target > 1) {
            if (maxDoubles == 0) {
                return ans + target - 1;
            }
            if (target % 2 == 0) {
                target /= 2;
                maxDoubles--;
                ans++;
            } else {
                target -= 1;
                ans++;
            }
        }
        return ans;
    }

    //[5982].解决智力问题
    public static long mostPoints(int[][] questions) {
        int n = questions.length;
        //以i到n-1的范围所获得最大分数
        long[] dp = new long[n];
        dp[n - 1] = questions[n - 1][0];
        for (int i = n - 2; i >= 0; i--) {
            int j = questions[i][1] + i + 1;
            //不做该题
            long selectNot = dp[i + 1];
            //做该题
            long select = j < n ? dp[j] + questions[i][0] : questions[i][0];
            dp[i] = Math.max(selectNot, select);
        }
        return dp[0];
    }

    //[2029].石子游戏 IX
    public boolean stoneGameIX(int[] stones) {
        int cnt0 = 0, cnt1 = 0, cnt2 = 0;
        for (int stone : stones) {
            if (stone % 3 == 0) cnt0++;
            else if (stone % 3 == 1) cnt1++;
            else if (stone % 3 == 2) cnt2++;
        }
        //11212121 => 1122, 12, 11222都是A能赢
        //22121212 => 2211, 21, 22111都是A能赢
        //偶数, A只要选择数量较小的必赢(前提1和2都不为0)
        if (cnt0 % 2 == 0) {
            return cnt1 >= 1 && cnt2 >= 1;
        }
        //奇数，相当于有一次先手交换，B一定是想先交换掉然后选择最优的情况
        //A只要选择较多的，去抵消一次，B要赢一定是较小数量的，所以A的数量超过2，较多的抵消之后，还会比较少的多1，导致A必赢
        return cnt1 - cnt2 > 2 || cnt2 - cnt1 > 2;
    }

    //[1345].跳跃游戏 IV
    public static int minJumps(int[] arr) {
        //通过bfs，将最近的元素入队，从距离最近开始，每遍历一层，距离+1，有三种选择，左边，右边，相同节点。对于所有节点，只有第一次赋值的时候才是最短距离
        //对于相同节点，需要用哈希表，遍历到该节点需要更新所有节点的距离为+1，然后就可以把该哈希表删除(已经是最短距离了，减少遍历次数)，优先选择最远的的，因为可能最后一个节点可以通过相同来实现最短距离跳跃。
        Map<Integer, List<Integer>> map = new HashMap<>();
        int n = arr.length;
        int[] dis = new int[n];
        Arrays.fill(dis, Integer.MAX_VALUE);
        for (int i = n - 1; i >= 0; i--) {
            List<Integer> list = map.getOrDefault(arr[i], new ArrayList<>());
            list.add(i);
            map.put(arr[i], list);
        }
        LinkedList<Integer> queue = new LinkedList<>();
        queue.offer(0);
        dis[0] = 0;
        while (!queue.isEmpty()) {
            int idx = queue.poll(), step = dis[idx];
            if (idx == n - 1) return step;
            int l = idx - 1, r = idx + 1;
            if (r < n && dis[r] == Integer.MAX_VALUE) {
                dis[r] = step + 1;
                queue.offer(r);
            }
            if (l >= 0 && dis[l] == Integer.MAX_VALUE) {
                dis[l] = step + 1;
                queue.offer(l);
            }
            List<Integer> same = map.getOrDefault(arr[idx], new ArrayList<>());
            for (int ne : same) {
                if (dis[ne] == Integer.MAX_VALUE) {
                    dis[ne] = step + 1;
                    queue.offer(ne);
                }
            }
            //加速，等值跳只需要第一次更新，后面可以用来剪枝。
            map.remove(arr[idx]);
        }
        return -1;
    }

    //[1332].删除回文子序列
    public int removePalindromeSub(String s) {
        int n = s.length();
        int l = 0, r = n - 1;
        while (l < r) {
            if (s.charAt(l++) != s.charAt(r--)) return 2;
        }
        return 1;
    }

    //[2034].股票价格波动
    public class StockPrice {

        int maxTimestamp;
        Map<Integer, Integer> timestampStock;
        TreeMap<Integer, Integer> priceCount;

        public StockPrice() {
            timestampStock = new HashMap<>();
            maxTimestamp = 0;
            priceCount = new TreeMap<>();
        }

        public void update(int timestamp, int price) {
            maxTimestamp = Math.max(maxTimestamp, price);
            int oldPrice = timestampStock.getOrDefault(timestamp, 0);
            if (oldPrice > 0) {
                priceCount.put(oldPrice, priceCount.getOrDefault(oldPrice, 0) - 1);
                if (priceCount.get(oldPrice) == 0) {
                    priceCount.remove(oldPrice);
                }
            }

            timestampStock.put(timestamp, price);
            priceCount.put(price, priceCount.getOrDefault(price, 0) + 1);
        }

        public int current() {
            return timestampStock.get(maxTimestamp);
        }

        public int maximum() {
            //需要获取最大的价格，需要有序能够直接获得，但是因为有修改，所以价格涉及到删除操作，考虑红黑树
            //又因为相同的价格，可能在不同的时间戳被设值，那么需要记录count值，更新操作的时候，如果老价格数量为0，那么就将老价格记录删除掉
            return priceCount.lastKey();
        }

        public int minimum() {
            return priceCount.firstKey();
        }
    }

    //[5989].元素计数
    public static int countElements(int[] nums) {
        TreeSet<Integer> set = new TreeSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int count = 0;
        for (int num : nums) {
            Integer ceiling = set.higher(num);
            Integer floor = set.lower(num);
            if (ceiling != null && floor != null) {
                count++;
            }
        }
        return count;
    }

    //[5991].按符号重排数组
    public int[] rearrangeArray(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        int l = 0, f = 1;
        for (int i = 0; i < n; i++) {
            if (nums[i] > 0) {
                res[l] = nums[i];
                l += 2;
            } else {
                res[f] = nums[i];
                f += 2;
            }
        }
        return res;
    }

    //[5990].找出数组中的所有孤独数字
    public List<Integer> findLonely(int[] nums) {
        HashMap<Integer, Integer> set = new HashMap<>();
        for (int num : nums) {
            set.put(num, set.getOrDefault(num, 0) + 1);
        }

        List<Integer> res = new ArrayList<>();
        for (int num : nums) {
            if (set.get(num) == 1 && !set.containsKey(num - 1) && !set.containsKey(num + 1)) {
                res.add(num);
            }
        }
        return res;
    }

    //[2045].到达目的地的第二短时间
    public int secondMinimum(int n, int[][] edges, int time, int change) {
        List<Integer>[] graph = new List[n + 1];
        for (int i = 0; i <= n; i++) {
            graph[i] = new ArrayList<Integer>();
        }
        //graph 中记录每个结点的出度和入度
        for (int[] edge : edges) {
            graph[edge[0]].add(edge[1]); //出度
            graph[edge[1]].add(edge[0]); //入度
        }

        // path[i][0] 表示从 1 到 i 的最短路长度，path[i][1] 表示从 1 到 i 的严格次短路长度
        int[][] path = new int[n + 1][2];
        for (int i = 0; i <= n; i++) {
            Arrays.fill(path[i], Integer.MAX_VALUE);
        }
        path[1][0] = 0;
        Queue<int[]> queue = new ArrayDeque<int[]>();
        queue.offer(new int[]{1, 0});
        while (path[n][1] == Integer.MAX_VALUE) {
            int[] arr = queue.poll();
            //cur表示当前结点，len表示到达当前结点需要走的总路程
            int cur = arr[0], len = arr[1];
            //计算到达相邻结点，走的最短总路程，和次短总路程
            for (int next : graph[cur]) {
                //更新到达相邻结点的最短总路程
                if (len + 1 < path[next][0]) {
                    path[next][0] = len + 1;
                    //这里更新完最短总路程后，为什么不进行比较
                    // 原path[next][0]与path[next][1]的大小，从而更新次短总路程？
                    //答：最短总路程总是先到达的，故不可能存在次短总路程先到达，然后最短总路程才到达
                    queue.offer(new int[]{next, len + 1}); //用于更新next结点相邻结点的最短总路程
                } else if (len + 1 > path[next][0] && len + 1 < path[next][1]) {
                    //更新次短总路程
                    path[next][1] = len + 1;
                    //用于更新next结点相邻结点的次短总路程
                    queue.offer(new int[]{next, len + 1});
                }
            }
        }

        int ret = 0;
        //计算走了 path[n][1] 步，共需要等待多少红灯和共需要多少时间
        for (int i = 0; i < path[n][1]; i++) {
            //经过 (2 * change) 灯由绿灯变成绿灯，并且维持 change 秒
            //如果 ret 不在该范围到达，就无法到达后立即出发，需要等红灯
            //等待时间为，一个 (2 * change) 周期，减去 到达时间
            if (ret % (2 * change) >= change) {
                ret = ret + (2 * change - ret % (2 * change));
            }
            ret = ret + time;
        }
        return ret;
    }

    //[524].通过删除字母匹配到字典里最长单词
    public String findLongestWord(String s, List<String> dictionary) {
        return null;
    }

    //[477].汉明距离总和
    public int totalHammingDistance(int[] nums) {
        return 0;
    }

    //[1688].比赛中的配对次数
    public int numberOfMatches(int n) {
        return n - 1;
    }

    //[393].UTF-8 编码验证
    public boolean validUtf8(int[] data) {
        //识别第一个是多少个连续的1，后面遇到10就数量--，遇到数量>0且不是10的return false
        return false;
    }

    //[401].二进制手表
    public List<String> readBinaryWatch(int turnedOn) {
        return null;
    }

    //[402].移掉 K 位数字
    public String removeKdigits(String num, int k) {
        //"1432219", k = 3
        //维护单调递增栈，后面发现前面的大，就把栈中的移除，保证后面是单调递增
        return null;
    }

    public static void main(String[] args) {
//        System.out.println(maxSlidingWindow(new int[]{1, 3, -1, -3, 5, 3, 6, 7}, 3));
////        System.out.println(mostPoints(new int[][]{{3, 2}, {4, 3}, {4, 4}, {2, 5}}));
////        System.out.println(mostPoints(new int[][]{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}}));
////        System.out.println(Arrays.toString(divideString("abcdefghij", 3, 'x')));
////        System.out.println(minMoves(5, 0));
////        System.out.println(minMoves(19, 2));
////        System.out.println(minMoves(10, 4));
////        System.out.println(new AllOfThem().wordCount(new String[]{"ant", "act", "tack"}, new String[]{"tack", "act", "acti"}));
////        System.out.println(Arrays.toString(new AllOfThem().executeInstructions(3, new int[]{0, 1}, "RRDDLU")));
////        System.out.println(new AllOfThem().permute(new int[]{1, 2, 3}));
////        System.out.println(new AllOfThem().permuteUnique(new int[]{1, 1, 2}));
////
////        System.out.println(new AllOfThem().findMinHeightTrees(2, new int[][]{{0, 1}}));
////        System.out.println(new AllOfThem().findMinHeightTrees(6, new int[][]{{3, 0}, {3, 1}, {3, 2}, {3, 4}, {5, 4}}));
////        System.out.println(new AllOfThem().numTrees(3));
////        System.out.println(new AllOfThem().generate(5));
////        System.out.println(new AllOfThem().getRow(4));
////        System.out.println(new AllOfThem().partition("aabc"));
////        System.out.println(new AllOfThem().convert("PAYPALISHIRING", 4));
////        System.out.println(new AllOfThem().convert("PAYPALISHIRING", 3));
////        System.out.println(new AllOfThem().convert("A", 1));
////
////        System.out.println(new AllOfThem().reverse(120));
////        System.out.println(new AllOfThem().reverse(-123));
////        System.out.println(new AllOfThem().singleNumber(new int[]{1, 2, 1}));
////        System.out.println(new AllOfThem().singleNumber(new int[]{4, 1, 2, 1, 2}));
////        System.out.println(new AllOfThem().singleNumber2(new int[]{0, 1, 0, 1, 0, 1, -99}));
////        ListNode list = new ListNode(1);
////        list.next = new ListNode(2);
////        list.next.next = new ListNode(3);
////        list.next.next.next = new ListNode(4);
////        ListNode r1 = new AllOfThem().swapPairs(list);
////
////        ListNode second = new ListNode(1);
////        second.next = new ListNode(2);
////        second.next.next = new ListNode(3);
////        ListNode result = new AllOfThem().mergeTwoList(list, second);
////
////        ListNode l61 = new ListNode(1);
////        l61.next = new ListNode(2);
////        l61.next.next = new ListNode(3);
////        l61.next.next.next = new ListNode(4);
////        l61.next.next.next.next = new ListNode(5);
////        ListNode r61 = new AllOfThem().rotateRight(l61, 2);
////
////        ListNode l141 = new ListNode(1);
////        l141.next = new ListNode(2);
////        l141.next.next = new ListNode(3);
////        l141.next.next.next = new ListNode(4);
//////        l141.next.next.next.next = new ListNode(5);
////        new AllOfThem().reorderList(l141);
////
////
////        ListNode l328 = new ListNode(1);
////        l328.next = new ListNode(2);
////        l328.next.next = new ListNode(3);
////        l328.next.next.next = new ListNode(4);
////        ListNode r328 = new AllOfThem().oddEvenList(l328);
////
////        System.out.println(new AllOfThem().reverseWords("  Bob    Loves  Alice   "));
////
////        System.out.println(new AllOfThem().findLeftIndex(new int[]{1}, 5));
////        System.out.println(new AllOfThem().searchInsert(new int[]{1}, 5));
////        System.out.println(new AllOfThem().search3(new int[]{1, 0, 1, 1, 1}, 0));
////
////        System.out.println(new AllOfThem().letterCombinations("23"));
////        System.out.println(new AllOfThem().combinationSum(new int[]{2, 3, 4, 7}, 7));
////        System.out.println(new AllOfThem().combinationSum2(new int[]{10, 1, 2, 7, 6, 1, 5}, 8));
////        System.out.println(new AllOfThem().combinationSum3(3, 9));
////        System.out.println(new AllOfThem().subsets(new int[]{1, 2, 3, 4}));
////        System.out.println(new AllOfThem().subsetsWithDup(new int[]{1, 2, 2}));
////        System.out.println(new AllOfThem().trap(new int[]{0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1}));
////        System.out.println(new AllOfThem().trap(new int[]{4, 2, 0, 3, 2, 5}));
////        System.out.println(Arrays.toString(new AllOfThem().nextGreaterElement(new int[]{4, 1, 2}, new int[]{1, 3, 4, 2})));
////        System.out.println(new AllOfThem().combine(4, 2));
////        System.out.println(new AllOfThem().solveNQueens(4));
//////        System.out.println(new AllOfThem().find132pattern(new int[]{1,2,3,4}));
//////        System.out.println(new AllOfThem().find132pattern(new int[]{3,1,4,2}));
////        System.out.println(new AllOfThem().find132pattern(new int[]{-1, 3, 2, 0}));
//////        System.out.println(new AllOfThem().find132pattern(new int[]{3, 5, 0, 3, 4}));
////        System.out.println(Arrays.toString(new AllOfThem().dailyTemperatures(new int[]{30, 60, 90})));
////        System.out.println(Arrays.toString(new AllOfThem().dailyTemperatures(new int[]{73, 74, 75, 71, 69, 72, 76, 73})));
////        System.out.println(Arrays.toString(new AllOfThem().dailyTemperatures(new int[]{73})));
////        System.out.println(new AllOfThem().findUnsortedSubarray(new int[]{2, 6, 4, 8, 10, 9, 15}));
////        System.out.println(new AllOfThem().findUnsortedSubarray(new int[]{1, 2, 3, 4}));
////        System.out.println(new AllOfThem().findUnsortedSubarray(new int[]{1, 2, 3, 3, 3}));
////        System.out.println(new AllOfThem().findUnsortedSubarray(new int[]{1}));
////        System.out.println(new AllOfThem().grayCode(3));
////        System.out.println(new AllOfThem().superPow(2, new int[]{1, 0}));
////        System.out.println(new AllOfThem().trailingZeroes(10000));
////
////        TreeNode t199 = new TreeNode(1);
////        t199.left = new TreeNode(2);
////        t199.right = new TreeNode(3);
////        t199.left.right = new TreeNode(5);
////        t199.right.left = new TreeNode(4);
////        System.out.println(new AllOfThem().rightSideView(t199));
////        System.out.println(new AllOfThem().merge(new int[][]{{1, 4}, {1, 4}}));
////
////        System.out.println(new AllOfThem().numSquares(12));
////
////        TopVotedCandidate candidate = new AllOfThem.TopVotedCandidate(new int[]{0, 1, 1, 0, 0, 1, 0}, new int[]{0, 5, 10, 15, 20, 25, 30});
////        System.out.println(candidate.q(3));
////        System.out.println(candidate.q(12));
////        System.out.println(candidate.q(25));
////        System.out.println(candidate.q(15));
////        System.out.println(candidate.q(24));
////        System.out.println(candidate.q(8));
////        System.out.println(new AllOfThem().jump(new int[]{2, 3, 1, 1, 4}));
////
////        System.out.println(new AllOfThem().simplifyPath("../"));
////        System.out.println(new AllOfThem().timeRequiredToBuy(new int[]{2, 3, 4, 1}, 1));
////        System.out.println(new AllOfThem().minEatingSpeed(new int[]{30, 11, 23, 4, 20}, 6));
////        System.out.println(new AllOfThem().shipWithinDays(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 5));
////        System.out.println(new AllOfThem().mySqrt(2147395599));
////
////        TreeNode t1712 = new TreeNode(4);
////        t1712.left = new TreeNode(2);
////        t1712.right = new TreeNode(5);
////        t1712.left.left = new TreeNode(1);
////        t1712.left.right = new TreeNode(3);
////        t1712.right.right = new TreeNode(6);
////        t1712.left.left.left = new TreeNode(0);
////
////        new AllOfThem().convertBiNode(t1712);
////
////        System.out.println(new AllOfThem().spiralOrder(new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}));
////        System.out.println(new AllOfThem().generateMatrix(3));
////
////        NumArray numArray = new NumArray(new int[]{-2, 0, 3, -5, 2, -1});
////        System.out.println(numArray.sumRange(0, 2));
////        System.out.println(numArray.sumRange(2, 5));
////        System.out.println(numArray.sumRange(0, 5));
////
////        System.out.println(Arrays.toString(new AllOfThem().corpFlightBookings(new int[][]{{1, 2, 10}, {2, 3, 20}, {2, 5, 25}}, 5)));
////        System.out.println(new AllOfThem().countBattleships(new char[][]{{'X', '.', '.'}, {'X', 'X', 'X'}, {'X', '.', '.'}}));
////        System.out.println(new AllOfThem().countBattleshipsV2(new char[][]{{'X', '.', '.'}, {'X', 'X', 'X'}, {'X', '.', '.'}}));
////        System.out.println(new AllOfThem().findRepeatedDnaSequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"));
////        System.out.println(new AllOfThem().decodeCiphertext("ch   ie   pr", 3));
////        System.out.println(new AllOfThem().decodeCiphertext("coding", 1));
////        System.out.println(new AllOfThem().getDescentPeriods(new int[]{12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 4, 3, 10, 9, 8, 7}));
////        System.out.println(new AllOfThem().kIncreasing(new int[]{5, 5, 5, 5, 4}, 1));
////        System.out.println(new AllOfThem().findLeftIndex2(new int[]{1, 2, 2, 4}, 2));
////        System.out.println(new AllOfThem().findLeftIndex2(new int[]{1, 2, 2, 4}, 3));
////        System.out.println(new AllOfThem().findRightIndexV2(new int[]{1, 2, 2, 4}, 2));
////        System.out.println(new AllOfThem().findRightIndexV2(new int[]{1, 2, 2, 4}, 3));
////        System.out.println(new AllOfThem().findRightIndexV3(new int[]{1, 2, 2, 4}, 2));
////        System.out.println(new AllOfThem().findRightIndexV3(new int[]{1, 2, 2, 4}, 3));
////        System.out.println(new AllOfThem().isAdditiveNumber("199100199"));
////
////        System.out.println(new AllOfThem().nthUglyNumber(10));
////
////        System.out.println(new AllOfThem().eventualSafeNodes(new int[][]{{1, 2}, {2, 3}, {5}, {0}, {5}, {}, {}}));
////        System.out.println(new AllOfThem().eventualSafeNodes(new int[][]{{1, 2, 3, 4}, {1, 2}, {3, 4}, {0, 4}, {}}));
////        System.out.println(new AllOfThem().invalidTransactions(new String[]{"alice,20,1220,mtv", "alice,20,1220,mtv"}));
////        System.out.println(new AllOfThem().countGoodSubstrings("xyzzaz"));
////        System.out.println(new AllOfThem().countGoodSubstrings("aababcabc"));
////        System.out.println(new AllOfThem().intToRoman(3));
////        System.out.println(new AllOfThem().longestCommonPrefix(new String[]{"flower", "flow", "flight"}));
////        System.out.println(new AllOfThem().longestCommonPrefix(new String[]{"dog", "racecar", "car"}));
////        System.out.println(new AllOfThem().countQuadruplets(new int[]{1, 2, 3, 6}));
////        System.out.println(new AllOfThem().countQuadruplets(new int[]{1, 1, 1, 3, 5}));
////        System.out.println(new AllOfThem().removeElement(new int[]{1}, 1));
////        System.out.println(new AllOfThem().removeElement(new int[]{1}, 0));
////        System.out.println(new AllOfThem().addBinary("11", "1"));
////        System.out.println(new AllOfThem().restoreIpAddresses("25525511135"));
////        System.out.println(new AllOfThem().restoreIpAddresses("101023"));
////        System.out.println(new AllOfThem().restoreIpAddresses("010010"));
////        System.out.println(new AllOfThem().findCircleNum(new int[][]{{1, 1, 0}, {1, 1, 0}, {0, 0, 1}}));
////        System.out.println(new AllOfThem().grayCode(3));
////        System.out.println(new AllOfThem().minSwaps(new int[]{0, 1, 0, 1, 1, 0, 0}));
////        System.out.println(new AllOfThem().minSwaps(new int[]{0, 1, 1, 1, 0, 0, 1, 1, 0}));
////        System.out.println(new AllOfThem().minSwaps(new int[]{1, 1, 0, 0, 1}));
////        System.out.println(new AllOfThem().isInterleave("aabcc", "dbbca", "aadbbcbcac"));
////        System.out.println(new AllOfThem().isInterleave("aabcc", "dbbca", "aadbbbaccc"));
////        System.out.println(new AllOfThem().isInterleave("", "", ""));
//        TreeNode t623 = new TreeNode(4);
//        t623.left = new TreeNode(2);
//        t623.left.left = new TreeNode(3);
//        t623.left.right = new TreeNode(1);
//        t623.right = new TreeNode(6);
//        t623.right.left = new TreeNode(5);
//
//        System.out.println(new AllOfThem().addOneRow(t623, 1, 2));

//        compress(new char[]{'a', 'a', 'b', 'b', 'c', 'c', 'c'});

//        System.out.println(new AllOfThem().diffWaysToCompute("2*3-4*5"));
//        System.out.println(findNthDigit(3));

//        System.out.println(lengthLongestPath("dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"));
//        System.out.println(lengthLongestPath("file1.txt\nfile2.txt\nlongfile.txt"));
    }
}
