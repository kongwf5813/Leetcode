package com.owen.algorithm.v3;

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
        public Node parent;
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

    //[4].寻找两个正序数组的中位数
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        int left = (m + n + 1) / 2;
        int right = (m + n + 2) / 2;
        return (getKth(nums1, 0, m - 1, nums2, 0, n - 1, left) +
                getKth(nums1, 0, m - 1, nums2, 0, n - 1, right)) * 0.5d;
    }

    public int getKth(int[] nums1, int s1, int e1, int[] nums2, int s2, int e2, int k) {
        int len1 = e1 - s1 + 1;
        int len2 = e2 - s2 + 1;
        if (len1 > len2) return getKth(nums2, s2, e2, nums1, s1, e1, k);
        if (len1 == 0) return nums2[s2 + k - 1];
        if (k == 1) return Math.min(nums1[s1], nums2[s2]);

        //排除k/2的数据
        //长度不够的时候，可以先取最小的
        int i = s1 + Math.min(len1, k / 2) - 1;
        int j = s2 + Math.min(len2, k / 2) - 1;

        if (nums1[i] > nums2[j]) {
            //排除掉s2～j的部分，注意k的运算是，排除掉了几个，扣多少
            return getKth(nums1, s1, e1, nums2, j + 1, e2, k - (j - s2 + 1));
        } else {
            //排除掉s1~i的部分
            return getKth(nums1, i + 1, e1, nums2, s2, e2, k - (i - s1 + 1));
        }
    }

    //[5].最长回文子串
    public String longestPalindrome(String s) {
        int n = s.length();
        if (n == 0) return "";

//        //注意只要字符串非空，一定有一个单字符必定是回文，默认最小值为1
//        int maxLen = 1, start = 0;
//        boolean[][] dp = new boolean[n][n];
//        for (int i = 0; i < n; i++) {
//            dp[i][i] = true;
//        }
//        for (int i = n - 1; i >= 0; i--) {
//            for (int j = i + 1; j < n; j++) {
//                dp[i][j] = s.charAt(i) == s.charAt(j) && (j - i < 3 || dp[i + 1][j - 1]);
//                if (dp[i][j] && j - i + 1 > maxLen) {
//                    maxLen = j - i + 1;
//                    start = i;
//                }
//            }
//        }
//        return s.substring(start, start + maxLen);

        int maxLen = 0, begin = 0;
        for (int i = 0; i < n; i++) {
            //奇数，两边扩散
            int len1 = expand(s, i, i);
            //偶数，两边扩散
            int len2 = expand(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > maxLen) {
                maxLen = len;
                //有偶数的情况，综合取len-1
                begin = i - (len - 1) / 2;
            }
        }
        return s.substring(begin, begin + maxLen);
    }

    private int expand(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        //不想等
        return right - left + 1 - 2;
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
        boolean begin = false;
        int n = s.length();
        int sign = 1, num = 0;
        for (int i = 0; i < n; i++) {
            char ch = s.charAt(i);
            if (!begin && ch == ' ') {
                continue;
            } else if (!begin && ch == '-') {
                sign = -1;
            } else if (!begin && ch == '+') {
                sign = 1;
            } else if (Character.isDigit(ch)) {
                int a = sign * (ch - '0');
                if (num < Integer.MIN_VALUE / 10 || (num == Integer.MIN_VALUE / 10 && a < Integer.MIN_VALUE % 10))
                    return Integer.MIN_VALUE;
                if (num > Integer.MAX_VALUE / 10 || (num == Integer.MAX_VALUE / 10 && a > Integer.MAX_VALUE % 10))
                    return Integer.MAX_VALUE;
                num = num * 10 + a;
            } else {
                break;
            }
            begin = true;
        }
        return num;
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

    //[10].正则表达式匹配
    public static boolean isMatch(String ss, String pp) {
        //技巧：往原字符头部插入空格，这样得到 char 数组是从 1 开始，而且可以使得 f[0][0] = true，可以将 true 这个结果滚动下去
        int m = ss.length(), n = pp.length();
        ss = " " + ss;
        pp = " " + pp;
        //f(i,j) 代表考虑 s 中的 1~i 字符和 p 中的 1~j 字符 是否匹配
        boolean[][] f = new boolean[m + 1][n + 1];
        f[0][0] = true;
        //p[j] != *  => f(i, j) = f(i-1, j-1) && (s[i] == p[j] || p[j] == '.')
        //p[j] == *  匹配0次 f(i, j) = f(i, j-2)    #a  #b*
        //p[j] == *  匹配1次 f(i, j) = f(i-1, j-2) && (s[i] = p[j-1] || p[j-1] == '.')
        //p[j] == *  匹配2次 f(i, j) = f(i-2, j-2) && (s[i-1:i] = p[j-1] || p[j-1] == '.')
        //f(i, j) = f(i, j-2) || f(i-1, j-2) && (s[i]匹配了p[j-1]) ||  f(i-2, j-2) && (s[i-1:i]匹配了p[j-1])
        //f(i-1, j) = f(i-1, j-2) || f(i-2, j-2) && (s[i-1]匹配了p[j-1]) ||  f(i-3, j-2) && (s[i-2:i-1]匹配了p[j-1])
        //f(i,j) = f(i, j-2) || f(i-1,j) && s[i]匹配了p[j-1]
        //f(i,j) = f(i, j-2) || f(i-1,j) && (s[i] == p[j-1] || p[j-1] == '.')
        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                //后面为*的，必须跳过，配合后面用
                if (j + 1 <= n && pp.charAt(j + 1) == '*') continue;
                if (pp.charAt(j) != '*') {
                    //对应了 p[j] 为普通字符和 '.' 的两种情况
                    f[i][j] = i >= 1 && f[i - 1][j - 1] && (ss.charAt(i) == pp.charAt(j) || pp.charAt(j) == '.');
                } else {
                    //对应了 p[j] 为 '*' 的情况，这边是通过数学归纳法总结出来的
                    f[i][j] = (j >= 2 && f[i][j - 2]) || (i >= 1 && f[i - 1][j] && (ss.charAt(i) == pp.charAt(j - 1) || pp.charAt(j - 1) == '.'));
                }
            }
        }
        return f[m][n];
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

    //[15].三数之和
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        List<List<Integer>> res = new ArrayList<>();
        //锚定第一个数之和，就是左右双指针，判断重复两个点： 第一个数第二次处理重复可以跳过，第二个数重复加速
        for (int i = 0; i < n - 2; i++) {
            if (i > 0 && nums[i - 1] == nums[i]) continue;
            int j = i + 1, k = n - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum == 0) {
                    res.add(Arrays.asList(nums[i], nums[j], nums[k]));
                    while (j < k && nums[j] == nums[++j]) ;
                    while (j < k && nums[k] == nums[--k]) ;
                } else if (sum > 0) {
                    while (j < k && nums[k] == nums[--k]) ;
                } else {
                    while (j < k && nums[j] == nums[++j]) ;
                }
            }
        }
        return res;
    }

    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int n = nums.length;
        int ans = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < n; i++) {
            // 保证和上一次枚举的元素不相等
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1, right = n - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (Math.abs(sum - target) < Math.abs(ans - target)) {
                    ans = sum;
                }
                if (ans == target) {
                    return ans;
                } else if (sum > target) {
                    right--;
                } else if (sum < target) {
                    left++;
                }
            }
        }
        return ans;
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
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode slow = dummy, fast = head;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
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
    public ListNode mergeTwoLists(ListNode first, ListNode second) {
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
        if (left == 0 && right == 0) {
            res.add(sb.toString());
            return;
        }
        if (left > right) return;

        if (left > 0) {
            sb.append('(');
            backtraceForGenerateParenthesis(left - 1, right, res, sb);
            sb.deleteCharAt(sb.length() - 1);
        }
        if (right > 0) {
            sb.append(')');
            backtraceForGenerateParenthesis(left, right - 1, res, sb);
            sb.deleteCharAt(sb.length() - 1);
        }
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
        //[head, p)的区间交换
        ListNode newHead = reverse(head, p);
        ListNode last = reverseKGroup(p, k);
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

    //[32].最长有效括号
    public int longestValidParentheses(String s) {
        int n = s.length();
        if (n == 0) return 0;
        //以i为结尾的最长有效括号长度
        //()                i == ) && i-1 == (  dp[i] = dp[i-2] +2;
        //)(())             i == ) && i-1 == )  && i - dp[i-1]-1 == (   dp[i] = dp[i - dp[i-1]-2] + dp[i-1]+2;
        //((())             i == ) && i-1 == )  && i - dp[i-1]-1 == (   dp[i] = dp[i-1]+2;
        int[] dp = new int[n];
        int maxAns = 0;
        for (int i = 1; i < n; i++) {
            if (s.charAt(i) == ')') {
                if (s.charAt(i - 1) == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                } else if (i > dp[i - 1] && s.charAt(i - dp[i - 1] - 1) == '(') {
                    dp[i] = dp[i - 1] + 2 + (i - dp[i - 1] - 2 > 0 ? dp[i - dp[i - 1] - 2] : 0);
                }
                maxAns = Math.max(maxAns, dp[i]);
            }
        }
        return maxAns;

//        Stack<Integer> stack = new Stack<>();
//        //最后一个没有匹配成功的右括号位置
//        stack.push(-1);
//        int maxAns = 0;
//        for (int i = 0; i < n; i++) {
//            char ch = s.charAt(i);
//            if (ch == '(') {
//                stack.push(i);
//            } else {
//                stack.pop();
        //本来遇到)就要把栈顶弹出去，发现空了，说明没有匹配
//                if (stack.isEmpty()) {
//                    //最后一个没有匹配成功的右括号位置
//                    stack.push(i);
//                } else {
        //说明匹配成功了，就更新位置，栈顶位置就是右括号没有匹配的位置
//                   maxAns = Math.max(maxAns, i - stack.peek());
//                }
//            }
//        }
//        return maxAns;
    }

    //[33].搜索旋转排序数组
    public int search(int[] nums, int target) {
        int n = nums.length;
        if (n == 0) return -1;
        if (n == 1) return nums[0] == target ? 0 : -1;
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            //相等，肯定返回
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[0] <= nums[mid]) {
                //[0, mid)有序递增
                if (nums[0] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                //(mid, n-1]有序递增
                if (nums[mid] < target && target <= nums[n - 1]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }

        }
        return -1;
    }

    //[34].在排序数组中查找元素的第一个和最后一个位置
    public int[] searchRange(int[] nums, int target) {
//        int leftBound = findIndex(nums, target, true);
//        int rightBound = findIndex(nums, target, false);
//        return new int[]{leftBound, rightBound};

        if (nums.length == 0) return new int[]{-1, -1};
        int left = leftBinarySearch(nums, target);
        int right = rightBinarySearch(nums, target);
        return new int[]{left, right};
    }

    private int leftBinarySearch(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                //左边界需要找到相等的值，左边界需要收缩
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return nums[left] == target ? left : -1;
    }

    private int rightBinarySearch(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left + 1) / 2;
            //右边界需要找到相等的值，右边界需要收缩
            if (nums[mid] > target) {
                right = mid - 1;
            } else {
                left = mid;
            }
        }
        return nums[left] == target ? left : -1;
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
                //i作为格子序数，每个格子有3行3列， j作为格子中的第几个
                //每一行有3个大格子，那么x的偏移就是垂直方向上i/3个格子，每个大格子还有3行，所以第一个元素的x值为(i/3) * 3
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
        //先排个序，然后去剪枝
        Arrays.sort(candidates);
        backtraceForCombinationSum(candidates, 0, 0, target, res, new LinkedList<>());
        return res;
    }

    private void backtraceForCombinationSum(int[] candidates, int idx, int sum, int target, List<List<Integer>> res, LinkedList<Integer> select) {
        //剪枝1
        if (sum > target) return;

        if (target == sum) {
            res.add(new ArrayList<>(select));
            return;
        }

        for (int i = idx; i < candidates.length; i++) {
            //后面的更大，剪枝2
            if (sum + candidates[i] > target) {
                break;
            }
            select.addLast(candidates[i]);
            backtraceForCombinationSum(candidates, i, sum + candidates[i], target, res, select);
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
        //剪枝1
        if (target < 0) {
            return;
        }
        if (target == 0) {
            res.add(new ArrayList<>(select));
            return;
        }

        for (int i = s; i < candidates.length; i++) {
            //剪枝2
            if (candidates[i] > target) {
                break;
            }
            //本层如果有重复的话，只选第一个，后面的直接跳过
            if (i > s && candidates[i] == candidates[i - 1]) {
                continue;
            }
            select.addLast(candidates[i]);
            //元素不能重复选择，只能从下一个选择
            backtraceForCombinationSum2(candidates, i + 1, target - candidates[i], res, select);
            select.removeLast();
        }
    }

    //[41].缺失的第一个正数
    public int firstMissingPositive(int[] nums) {
        //缺失的第一个正数必定在[1, n+1]之间, 而数组是从0到n
        //[1,2,0]  => nums[i] -1 作为新的index
        //[3,4,-1,1] => [1, -1, 3, 4]
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            //[1,1]这种情况会进入死循环，题目没限定重复的范围，如果索引位置上的数值不相等，替换成正确的
            while (nums[i] >= 1 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                int temp = nums[nums[i] - 1];
                nums[nums[i] - 1] = nums[i];
                nums[i] = temp;
            }
        }

        for (int i = 0; i < n; i++) {
            if (nums[i] - 1 != i) {
                return i + 1;
            }
        }
        return n + 1;
    }

    //[42].接雨水
    public int trap(int[] height) {
//        //找某侧最近一个比其大的值，使用单调栈维持栈内元素递减；
//        //找某侧最近一个比其小的值，使用单调栈维持栈内元素递增
//        int res = 0;
//        Stack<Integer> stack = new Stack<>();
//        for (int i = 0; i < height.length; i++) {
//            //   |   |
//            //   | | |
//            //   l c r， 维护单调递减栈，当大元素进栈，会压栈，弹出小的元素，那么此时栈顶一定大于弹出的元素
//            while (!stack.isEmpty() && height[i] > height[stack.peek()]) {
//                int cur = stack.pop();
//                // 如果栈内没有元素，说明当前位置左边没有比其高的柱子，跳过
//                if (stack.isEmpty()) {
//                    continue;
//                }
//                // 左右位置，并有左右位置得出「宽度」和「高度」
//                int l = stack.peek(), r = i;
//                int w = r - l + 1 - 2;
//                int h = Math.min(height[l], height[r]) - height[cur];
//                res += w * h;
//            }
//            stack.push(i);
//        }
//        return res;

        //双指针解法最优
        int n = height.length;
        int leftMax = 0, rightMax = 0;
        int left = 0, right = n - 1;
        int ans = 0;
        while (left <= right) {
            leftMax = Math.max(leftMax, height[left]);
            rightMax = Math.max(rightMax, height[right]);
            if (leftMax < rightMax) {
                ans += leftMax - height[left];
                left++;
            } else {
                ans += rightMax - height[right];
                right--;
            }
        }
        return ans;
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

    //[44].通配符匹配
    public boolean isMatch2(String s, String p) {
        //当为?时，f(i,j) = f(i-1,j-1)
        //                 匹配空格       匹配前面一个字符    匹配前面两个字符
        //当为*时，f(i,j) = f(i, j-1) || f(i-1, j-1) || f(i-2, j-1) || ... || f(i-k, j-1) (i>=k)
        //当为*时，f(i-1,j) =            f(i-1, j-1) || f(i-2, j-1) || f(i-3, j-1) || ...
        //f(i,j) = f(i, j-1) || f(i-1, j)
        int m = s.length(), n = p.length();
        s = " " + s;
        p = " " + p;
        // f(i,j) 代表考虑 s 中的 1~i 字符和 p 中的 1~j 字符 是否匹配
        boolean[][] f = new boolean[m + 1][n + 1];
        f[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j) != '*') {
                    f[i][j] = i >= 1 && f[i - 1][j - 1] && (s.charAt(i) == p.charAt(j) || p.charAt(j) == '?');
                } else {
                    f[i][j] = f[i][j - 1] || (i >= 1 && f[i - 1][j]);
                }
            }
        }
        return f[m][n];
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
        int n = nums.length;
        List<List<Integer>> res = new ArrayList<>();
        backtraceForPermute(nums, new boolean[n], res, new LinkedList<>());
        return res;
    }

    private void backtraceForPermute(int[] nums, boolean[] visited, List<List<Integer>> res, LinkedList<Integer> path) {
        if (path.size() == nums.length) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i]) {
                continue;
            }
            visited[i] = true;
            path.addLast(nums[i]);
            backtraceForPermute(nums, visited, res, path);
            path.removeLast();
            visited[i] = false;
        }
    }

    //[47].全排列 II
    public List<List<Integer>> permuteUnique(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        backtraceForPermuteUnique(nums, new boolean[n], new LinkedList<>(), res);
        return res;
    }

    private void backtraceForPermuteUnique(int[] nums, boolean[] visited, LinkedList<Integer> path, List<List<Integer>> res) {
        if (path.size() == nums.length) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            if (visited[i]) continue;
            //决策树画完之后，发现01这种状态需要剪枝，意思是重复的数。
            //一定从左边往右边选: 如果左边的还没有选，则右边的也不选，直接跳过。
            if (i > 0 && nums[i] == nums[i - 1] && !visited[i - 1]) continue;
            path.addLast(nums[i]);
            visited[i] = true;
            backtraceForPermuteUnique(nums, visited, path, res);
            visited[i] = false;
            path.removeLast();
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

    //[60].排列序列
    public String getPermutation(int n, int k) {
        int[] frac = new int[n + 1];
        List<Integer> select = new ArrayList<>();
        frac[0] = 1;
        for (int i = 1; i <= n; i++) {
            frac[i] = frac[i - 1] * i;
            select.add(i);
        }
        k--;
        StringBuilder sb = new StringBuilder();
        for (int j = n - 1; j >= 0; j--) {
            int index = k / frac[j];
            sb.append(select.remove(index));
            k %= frac[j];
        }
        return sb.toString();
    }

    //[61].旋转链表
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || head.next == null || k == 0) return head;
        int count = 0;
        ListNode p = head;
        while (p != null) {
            p = p.next;
            count++;
        }
        k %= count;
        if (k == 0) return head;

        ListNode fast = head;
        for (int i = 0; i < k; i++) {
            fast = fast.next;
        }
        ListNode slow = head;
        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }
        ListNode newHead = slow;
        slow.next = null;
        fast.next = head;
        return newHead;
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

    //[72].编辑距离
    public int minDistance0(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        //word1长度为i，word2长度为j的最小编辑距离
        int[][] dp = new int[m + 1][n + 1];
        //初始值
        for (int i = 1; i <= m; i++) dp[i][0] = i;
        for (int j = 1; j <= n; j++) dp[0][j] = j;

        //当前值依赖左上方，左边，上边的值，求得值为dp[m][n]，由短及远，所以遍历方向是从(0,0)从左到右，从上到下
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                //这边是索引
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    //hab hae => dp[i][j] 还依赖dp[i-1][j-1] +1
                    dp[i][j] = Math.min(dp[i - 1][j], Math.min(dp[i][j - 1], dp[i - 1][j - 1])) + 1;
                }
            }
        }
        return dp[m][n];
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

//        int p0 = 0, n = nums.length;
//        for (int i = 0; i < n; i++) {
//            if (nums[i] == 0) {
//                int temp = nums[i];
//                nums[i] = nums[p0];
//                nums[p0++] = temp;
//            }
//        }
//        int p1 = p0;
//        for (int i = p1; i < n; i++) {
//            if (nums[i] == 1) {
//                int temp = nums[i];
//                nums[i] = nums[p1];
//                nums[p1++] = temp;
//            }
//        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    //[76].最小覆盖子串
    public static String minWindow(String s, String t) {
        int m = s.length(), n = t.length();
        if (m < n) return "";
        int[] need = new int[58];
        for (int i = 0; i < n; i++) {
            need[t.charAt(i) - 'A']++;
        }
        int[] window = new int[58];
        int minLen = Integer.MAX_VALUE;
        String res = "";
        for (int l = 0, r = 0; r < m; r++) {
            window[s.charAt(r) - 'A']++;

            //缩窗口
            while (checkForMinWindow(window, need)) {
                if (minLen > r - l + 1) {
                    minLen = r - l + 1;
                    res = s.substring(l, r + 1);
                }
                window[s.charAt(l) - 'A']--;
                l++;
            }
        }
        return res;
    }

    private static boolean checkForMinWindow(int[] src, int[] target) {
        for (int i = 0; i < 58; i++) {
            if (src[i] < target[i]) return false;
        }
        return true;
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

    //[84]. 柱状图中最大的矩形
    public int largestRectangleArea(int[] heights) {
        int[] newHeights = new int[heights.length + 2];
        //跟接雨水的问题类似，雨水是高低高，这题是低高低
        for (int i = 0; i < heights.length; i++) {
            newHeights[i + 1] = heights[i];
        }

        int res = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < newHeights.length; i++) {
            while (!stack.isEmpty() && newHeights[i] < newHeights[stack.peek()]) {
                int cur = stack.pop();
                res = Math.max((i - stack.peek() - 1) * newHeights[cur], res);
            }
            stack.push(i);
        }
        return res;
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
//        boolean[] visit = new boolean[1 << n];
//        LinkedList<Integer> select = new LinkedList<>();
//        backtraceForGrayCode(n, 0, visit, select);
//        return select;
        //镜像构造法
        //0 0  00
        //  1  01
        //     11
        //     10
        List<Integer> res = new ArrayList<Integer>() {{
            add(0);
        }};
        int head = 1;
        for (int i = 0; i < n; i++) {
            int size = res.size();
            for (int j = size - 1; j >= 0; j--) {
                res.add(head + res.get(j));
            }
            head <<= 1;
        }
        return res;
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

    //[91].解码方法
    public int numDecodings(String s) {
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;
        //长度为i的字符串的方法数
        for (int i = 1; i <= n; i++) {
            int a = s.charAt(i - 1) - '0';
            //单独一个数字
            if (a >= 1 && a <= 9) {
                dp[i] = dp[i - 1];
            }
            //可以和前面的组成一个数字
            if (i > 1) {
                int b = (s.charAt(i - 2) - '0') * 10 + (s.charAt(i - 1) - '0');
                if (b >= 10 && b <= 26) {
                    dp[i] += dp[i - 2];
                }
            }
        }
        return dp[n];
    }

    //[92].反转链表 II
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode pre = dummy;
        for (int i = 0; i < left - 1; i++) {
            pre = pre.next;
        }

        //pre表示前面的节点，后面的节点为新的头节点
        //cur表示当前节点，需要链接到最后面的节点
        //next表示后置节点，需要连接到pre.next(新头)
        //头插法
        ListNode cur = pre.next;
        for (int i = 0; i < right - left; i++) {
            ListNode next = cur.next;
            //cur连接到最后的节点
            cur.next = next.next;
            //翻转后置节点，不是cur节点，而是pre.next
            next.next = pre.next;
            //尾结点变更成后置节点
            pre.next = next;
        }
        return dummy.next;
    }

    //[92].反转链表 II（递归）
    public ListNode reverseBetweenV2(ListNode head, int left, int right) {
        if (left == 1) {
            return reverseN(head, right);
        }
        head.next = reverseBetweenV2(head.next, left - 1, right - 1);
        return head;
    }

    private ListNode successor = null;

    private ListNode reverseN(ListNode head, int n) {
        if (n == 1) {
            successor = head.next;
            return head;
        }

        ListNode last = reverseN(head.next, n - 1);

        head.next.next = head;
        head.next = successor;
        return last;
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
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            res.add(cur.val);
            cur = cur.right;
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
    //[剑指 Offer 07].重建二叉树
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
    ListNode p;

    public TreeNode sortedListToBST(ListNode head) {
//        if (head == null) return null;
//        ListNode slow = head, fast = head, pre = null;
//        while (fast != null && fast.next != null) {
//            pre = slow;
//            slow = slow.next;
//            fast = fast.next.next;
//        }
//        TreeNode root = new TreeNode(slow.val);
//        //只有一个节点了
//        if (pre == null) return root;
//
//        //断开前面一个节点
//        pre.next = null;
//        root.left = sortedListToBST(head);
//        root.right = sortedListToBST(slow.next);
//        return root;

        //法二，效率比较低，每次都需要找中点
//        return buildTree(head, null);

        //法三，利用中序遍历，延迟找中点
        p = head;
        int len = getLength(head);
        return buildTree(0, len - 1);
    }

    private TreeNode buildTree(ListNode left, ListNode right) {
        if (left == right) {
            return null;
        }
        ListNode mid = getMedian(left, right);
        TreeNode root = new TreeNode(mid.val);
        root.left = buildTree(left, mid);
        root.right = buildTree(mid.next, right);
        return root;
    }

    private ListNode getMedian(ListNode left, ListNode right) {
        ListNode slow = left, fast = left;
        while (fast != right && fast.next != right) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    private int getLength(ListNode head) {
        int ans = 0;
        while (head != null) {
            ans++;
            head = head.next;
        }
        return ans;
    }

    private TreeNode buildTree(int left, int right) {
        if (left > right) return null;
        int mid = left + (right - left) / 2;
        TreeNode root = new TreeNode(-1);
        root.left = buildTree(left, mid - 1);
        //左根右特性，这边第一次访问一定是中点。
        root.val = p.val;
        p = p.next;
        root.right = buildTree(mid + 1, right);
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
//        if (root == null) return;
//
//        //前序遍历
//        path.addLast(root.val);
//        if (root.left == null && root.right == null && targetSum == root.val) {
//            res.add(new ArrayList<>(path));
//            //回溯撤销节点的，加了return，会导致叶子节点会有撤销成功，导致路径上少减少一次撤销，从而使得下一次的选择会多一个节点。
//            //主要取决于前序遍历顺序不能变更。
//        }
//
//        dfsForPathSum(root.left, targetSum - root.val, path, res);
//        dfsForPathSum(root.right, targetSum - root.val, path, res);
//        path.removeLast();

        //如果在内部做节点回溯，那么中间不能有额外的return，否则导致节点会减少。而且回溯状态是等左右子树都结束了，回溯的状态其实是该节点。
        //如果在选择子节点的时候，回溯的状态其实是子节点。
        path.addLast(root.val);
        if (root.left == null && root.right == null && targetSum == root.val) {
            res.add(new ArrayList<>(path));
            return;
        }

        if (root.left != null) {
            dfsForPathSum(root.left, targetSum - root.val, path, res);
            path.removeLast();
        }

        if (root.right != null) {
            dfsForPathSum(root.right, targetSum - root.val, path, res);
            path.removeLast();
        }

    }

    //[114].二叉树展开为链表 (后序递归)
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

    //[114].二叉树展开为链表 (前序递归)
    TreeNode previous = null;

    private void flattenV2(TreeNode root) {
        if (root == null) return;
        if (previous != null) {
            previous.right = root;
            //前序遍历的时候更新掉左边的链接
            previous.left = null;
        }
        previous = root;
        flatten(root.left);
        flatten(root.right);
    }

    //[115].不同的子序列
    public int numDistinct(String s, String t) {
        int m = s.length(), n = t.length();
        //长度为i的s和长度为j的t，出现的个数
        int[][] dp = new int[m + 1][n + 1];
        //当s长度为0的时候，dp[0][i] = 0
        //当t长度为0的时候，dp[i][0] = 1, 因为空字符串可以通过s删掉得到，有且仅有唯一一种办法可以得到。
        for (int i = 0; i < m; i++) dp[i][0] = 1;

        //bag  bag, bagg bag, 可以对s进行删除操作
        //s[i] == t[j] => dp[i][j] = dp[i-1][j-1] + dp[i-1][j];
        //s[i] != t[j] => dp[i][j] = dp[i-1][j];

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[m][n];
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
//        int n = triangle.size();
//        //走到(i, j)点的最小路径和
//        int[][] dp = new int[n][n];
//        dp[0][0] = triangle.get(0).get(0);
//        for (int i = 1; i < n; i++) {
//            for (int j = 0; j <= i; j++) {
//                if (j == 0) {
//                    dp[i][j] = dp[i - 1][j] + triangle.get(i).get(j);
//                } else if (j == i) {
//                    dp[i][j] = dp[i - 1][j - 1] + triangle.get(i).get(j);
//                } else {
//                    dp[i][j] = Math.min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle.get(i).get(j);
//                }
//            }
//        }
//        int min = dp[n - 1][0];
//        for (int i = 1; i < n; i++) {
//            min = Math.min(min, dp[n - 1][i]);
//        }
//        return min;

        //空间压缩
//        int n = triangle.size();
//        //到底层i的最短路径和
//        int[] dp = new int[n];
//        dp[0] = triangle.get(0).get(0);
//        int pre = 0, cur;
//        //  pre          cur, pre'     cur'
//        // (i-1, j-1)   (i-1, j)     (i-1, j+1)
//        //        ＼        ↓    ＼      ↓
//        //               (i, j)       (i, j+1)
//        for (int i = 1; i < n; i++) {
//            for (int j = 0; j <= i; j++) {
//                cur = dp[j];
//                if (j == 0) {
//                    dp[j] = cur + triangle.get(i).get(j);
//                } else if (j == i) {
//                    dp[j] = pre + triangle.get(i).get(j);
//                } else {
//                    dp[j] = Math.min(pre, cur) + triangle.get(i).get(j);
//                }
//                pre = cur;
//            }
//        }
//        int min = dp[0];
//        for (int i = 1; i < n; i++) {
//            min = Math.min(min, dp[i]);
//        }
//        return min;
        int n = triangle.size();
        int[] dp = new int[n];
        dp[0] = triangle.get(0).get(0);
        for (int i = 1; i < n; i++) {
            //倒序遍历就不需要引入变量
            for (int j = i; j >= 0; j--) {
                if (j == 0) {
                    dp[j] = dp[j] + triangle.get(i).get(j);
                } else if (j == i) {
                    dp[j] = dp[j - 1] + triangle.get(i).get(j);
                } else {
                    dp[j] = Math.min(dp[j - 1], dp[j]) + triangle.get(i).get(j);
                }
            }
        }
        int ans = Integer.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            ans = Math.min(ans, dp[i]);
        }
        return ans;
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

    //[124].二叉树中的最大路径和
    public int maxPathSum(TreeNode root) {
        AtomicInteger res = new AtomicInteger(Integer.MIN_VALUE);
        oneSideMax(root, res);
        return res.get();
    }

    private int oneSideMax(TreeNode root, AtomicInteger maxPathSum) {
        if (root == null) return 0;

        //只有单边路径和为正值才会选择
        int leftMax = Math.max(oneSideMax(root.left, maxPathSum), 0);
        int rightMax = Math.max(oneSideMax(root.right, maxPathSum), 0);

        int pathSum = leftMax + rightMax + root.val;
        if (maxPathSum.get() < pathSum) {
            maxPathSum.set(pathSum);
        }

        //单边路径和等于左右两边的最大和+根节点的值
        return Math.max(leftMax, rightMax) + root.val;
    }

    //[127].单词接龙
    public static int ladderLength(String beginWord, String endWord, List<String> wordList) {
        if (!wordList.contains(endWord)) return 0;
        Set<String> q1 = new HashSet<>();
        Set<String> q2 = new HashSet<>();
        q1.add(beginWord);
        q2.add(endWord);
        Set<String> visited = new HashSet<>();
        int step = 1;
        while (!q1.isEmpty() && !q2.isEmpty()) {
            Set<String> temp = new HashSet<>();
            for (String cur : q1) {
                if (q2.contains(cur)) return step;
                //这边设置也可以
                visited.add(cur);
                for (String word : wordList) {
                    if (!visited.contains(word) && canConvert(cur, word)) {
                        temp.add(word);
                    }
                }
            }
            step++;
            q1 = q2;
            q2 = temp;
        }
        return 0;
    }

    private static boolean canConvert(String cur, String word) {
        int diff = 0;
        for (int i = 0; i < cur.length(); i++) {
            if (cur.charAt(i) != word.charAt(i)) diff++;
        }
        return diff == 1;
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
        class Node {
            int key;
            int value;
            Node prev;
            Node next;

            public Node(int key, int value) {
                this.key = key;
                this.value = value;
            }
        }

        int capacity;
        Node head, tail;
        Map<Integer, Node> map;

        public LRUCache(int capacity) {
            head = new Node(-1, -1);
            tail = new Node(-1, -1);
            head.next = tail;
            tail.prev = head;
            this.capacity = capacity;
            this.map = new HashMap<>();
        }

        public int get(int key) {
            Node node = map.get(key);
            if (node == null) return -1;

            moveToHead(node);
            return node.value;
        }

        public void put(int key, int value) {
            Node node = map.get(key);
            if (node == null) {
                node = new Node(key, value);
                map.put(key, node);
                addHead(node);
                if (map.size() > capacity) {
                    Node delete = tail.prev;
                    remove(delete);
                    map.remove(delete.key);
                }
            } else {
                node.value = value;
                moveToHead(node);
            }
        }

        private void remove(Node node) {
            node.prev.next = node.next;
            node.next.prev = node.prev;
        }

        private void moveToHead(Node node) {
            remove(node);
            addHead(node);
        }

        private void addHead(Node node) {
            node.next = head.next;
            head.next = node;

            node.next.prev = node;
            node.prev = head;
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
//        StringBuilder sb = new StringBuilder();
//        int len = s.length();
//        int left = 0;
//        while (s.charAt(left) == ' ') {
//            left++;
//        }
//
//        for (int i = len - 1; i >= left; i--) {
//            int j = i;
//            while (i >= left && s.charAt(i) != ' ') {
//                i--;
//            }
//
//            if (i != j) {
//                sb.append(s.substring(i + 1, j + 1));
//                if (i > left) {
//                    sb.append(" ");
//                }
//            }
//        }
//        return sb.toString();

        s = s.trim();
        StringBuilder sb = new StringBuilder();
        for (int i = s.length() - 1; i >= 0; ) {
            int j = i;
            while (i >= 0 && s.charAt(i) != ' ') i--;
            sb.append(s.substring(i + 1, j + 1)).append(' ');
            while (i >= 0 && s.charAt(i) == ' ') i--;
        }
        return sb.deleteCharAt(sb.length() - 1).toString();
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
//        int left = 0, right = nums.length - 1;
//        while (left <= right) {
//            int mid = left + (right - left) / 2;
//            if (nums[mid] > nums[right]) {
//                left = mid + 1;
//            } else {
//                right = mid - 1;
//            }
//        }
//        return nums[left];

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

    //[154].寻找旋转排序数组中的最小值 II
    public int findMin2(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] > nums[r]) {
                l = mid + 1;
            } else if (nums[mid] < nums[r]) {
                //有可能是，所以r = mid;
                r = mid;
            } else {
                //砍掉右边界
                r--;
            }
        }
        return nums[l];
    }

    //[160].相交链表
    //[剑指 Offer 52].两个链表的第一个公共节点
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

    //[168].Excel表列名称
    public static String convertToTitle(int columnNumber) {
        StringBuilder sb = new StringBuilder();
        while (columnNumber > 0) {
            //26进制范围[0, 25]，遇到26进行进位，但是本题范围是1~26，所以要偏移1位之后进行计算
            columnNumber--;

            sb.append((char) (columnNumber % 26 + 'A'));
            columnNumber /= 26;
        }
        return sb.reverse().toString();
    }

    //[169].多数元素
    public int majorityElement(int[] nums) {
        int candidate = 0;
        int count = 0;
        for (int num : nums) {
            if (count != 0 && candidate == num) count++;
            else if (count == 0) {
                candidate = num;
                count = 1;
            } else {
                count--;
            }
        }
        count = 0;
        for (int num : nums) {
            if (candidate == num) count++;
        }
        return count > nums.length / 2 ? candidate : -1;
    }

    //[171].Excel 表列序号
    public int titleToNumber(String columnTitle) {
        int len = columnTitle.length();
        int res = 0;
        //AAB
        for (int i = 0; i < len; i++) {
            res = res * 26 + (columnTitle.charAt(i) - 'A' + 1);
        }
        return res;
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
            pushLeft(root);
        }

        public int next() {
            TreeNode root = stack.pop();
            //右子树作为当前节点，继续压栈
            pushLeft(root.right);
            return root.val;
        }

        public boolean hasNext() {
            return !stack.isEmpty();
        }

        private void pushLeft(TreeNode root) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
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

//        Map<Character, Integer> map = new HashMap<>();
//        map.put('A', 0);
//        map.put('C', 1);
//        map.put('T', 2);
//        map.put('G', 3);
//        //前10位先构造出来，一共需要20位，理论上够的
//        int x = 0;
//        for (int i = 0; i < 9; i++) {
//            x = (x << 2) | map.get(s.charAt(i));
//        }
//        int n = s.length();
//        Map<Integer, Integer> count = new HashMap<>();
//        for (int i = 0; i <= n - 10; i++) {
//            //前面算了9位，从index 9开始算
//            x = ((x << 2) | map.get(s.charAt(i + 9))) & ((1 << 20) - 1);
//            count.put(x, count.getOrDefault(x, 0) + 1);
//            if (count.get(x) == 2) {
//                res.add(s.substring(i, i + 10));
//            }
//        }
//        return res;
        int n = s.length();
        int N = (int) 1e5 + 10, P = 131313;
        int[] h = new int[N];
        int[] p = new int[N];
        p[0] = 1;
        for (int i = 1; i <= n; i++) {
            h[i] = h[i - 1] * P + s.charAt(i - 1);
            p[i] = p[i - 1] * P;
        }

        Map<Integer, Integer> count = new HashMap<>();
        for (int i = 1; i + 10 - 1 <= n; i++) {
            int j = i + 10 - 1;
            int hash = h[j] - h[i - 1] * p[j - i + 1];
            count.put(hash, count.getOrDefault(hash, 0) + 1);
            if (count.get(hash) == 2) {
                res.add(s.substring(i - 1, j));
            }
        }
        return res;

    }

    //[188].买卖股票的最佳时机 IV
    public int maxProfit(int k, int[] prices) {
        int n = prices.length;
        if (n == 0) return 0;
        //如果k的值超过n/2，就是无限次数，跟k没关系了
        if (k > n / 2) {
            int dp_i_0 = 0, dp_i_1 = -prices[0];
            for (int i = 1; i < n; i++) {
                int temp_i_0 = dp_i_0;
                int temp_i_1 = dp_i_1;
                dp_i_0 = Math.max(temp_i_0, temp_i_1 + prices[i]);
                dp_i_1 = Math.max(temp_i_1, temp_i_0 - prices[i]);
            }
            return dp_i_0;
        }

        int[][][] dp = new int[n][k + 1][2];
        for (int i = 1; i <= k; i++) {
            dp[0][i][0] = 0;
            dp[0][i][1] = -prices[0];
        }
        for (int i = 1; i < n; i++) {
            for (int j = 1; j <= k; j++) {
                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
                dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i]);
            }
        }
        return dp[n - 1][k][0];
    }

    //[189].轮转数组
    public void rotate(int[] nums, int k) {
//        int n = nums.length;
//        k = k % n;
//        //控制翻转的数量就可以
//        int count = 0;
//        for (int start = 0; start < n && count < n; start++) {
//            int current = start;
//            int pre = nums[current];
//            do {
//                count++;
//                int next = (current + k) % n;
//                int temp = nums[next];
//                nums[next] = pre;
//
//                current = next;
//                pre = temp;
//            } while (start != current);
//        }
        int n = nums.length;
        k = k % n;
        reverse(nums, 0, n - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, n - 1);
    }

    private void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
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
//        迭代
//        ListNode dummy = new ListNode(-1);
//        while (head != null) {
//            ListNode next = head.next;
//            head.next = dummy.next;
//            dummy.next = head;
//            head = next;
//        }
//        return dummy.next;

        //递归
        if (head == null || head.next == null) return head;
        ListNode last = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return last;
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
//        //2,3,1,2,4, 3
//        //2 5 6 8 12 15
//        int n = nums.length;
//        int[] preSum = new int[n + 1];
//        int sum = 0;
//        for (int i = 0; i < n; i++) {
//            sum += nums[i];
//            preSum[i + 1] = sum;
//        }
//        int left = 0, right = 0, res = Integer.MAX_VALUE;
//        while (right < nums.length) {
//            right++;
//            //找到合理的，就缩
//            while (preSum[right] - preSum[left] >= target) {
//                res = Math.min(res, right - left);
//                left++;
//            }
//
//        }
//        return res == Integer.MAX_VALUE ? 0 : res;

        //前缀和+滑动窗口，因为窗口是一个个往后扩，所以前缀和可以用给一个变量控制，而不用提前计算。
        int n = nums.length;
        if (n == 0) {
            return 0;
        }
        int sum = 0;
        int ans = Integer.MAX_VALUE;
        for (int l = 0, r = 0; r < n; r++) {
            sum += nums[r];
            while (sum >= target) {
                ans = Math.min(ans, r - l + 1);
                sum -= nums[l++];
            }
        }
        return ans == Integer.MAX_VALUE ? 0 : ans;
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
//        //第K大意味着是从小到大是第n-k位
//        return quickSort(nums, 0, nums.length - 1, nums.length - k);

        //堆排序，手撕大根堆
        int heapSize = nums.length;
        buildHeap(nums, heapSize);
        //每次交换之后就会调整一次最大值，所以k-1次出堆就可以
        for (int i = heapSize - 1; k > 1; i--, k--) {
            swap(nums, i, 0);
            maxHeapify(nums, 0, i);
        }
        return nums[0];
    }


    private void buildHeap(int[] nums, int heapSize) {
        //从非叶子节点开始调整堆
        for (int i = heapSize / 2; i >= 0; --i) {
            maxHeapify(nums, i, heapSize);
        }
    }

    private void maxHeapify(int[] nums, int i, int heapSize) {
        int l = 2 * i + 1, r = 2 * i + 2;
        int largest = i;
        if (l < heapSize && nums[l] > nums[largest]) {
            largest = l;
        }
        if (r < heapSize && nums[r] > nums[largest]) {
            largest = r;
        }

        if (largest != i) {
            swap(nums, i, largest);
            maxHeapify(nums, largest, heapSize);
        }
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

    //[224].基本计算器
    public int calculate(String s) {
        //法一：
//        return dfsForCalculate(s, 0)[0];
        //法二：
//        Deque<Character> queue = new ArrayDeque<>();
//        for (char ch : s.toCharArray()) {
//            queue.offer(ch);
//        }
//        return dfsForCalculate(queue);

        //法三：宫水三叶
        Map<Character, Integer> map = new HashMap<Character, Integer>() {{
            put('-', 1);
            put('+', 1);
            put('*', 2);
            put('/', 2);
        }};
        Stack<Integer> nums = new Stack<>();
        Stack<Character> ops = new Stack<>();
        nums.push(0);
        int n = s.length();
        char[] cs = s.toCharArray();
        for (int i = 0; i < n; i++) {
            char ch = cs[i];
            if (ch == ' ') {
                continue;
            } else if (ch == '(') {
                ops.push(ch);
            } else if (ch == ')') {
                while (!ops.isEmpty()) {
                    if (ops.peek() != '(') {
                        calc(nums, ops);
                    } else {
                        break;
                    }
                }
            } else if (Character.isDigit(ch)) {
                int j = i, num = 0;
                while (j < n && Character.isDigit(cs[j])) {
                    num = num * 10 + (cs[j] - '0');
                    j++;
                }
                nums.push(num);
                i = j - 1;
            } else {
                if (i > 0 && (cs[i - 1] == '(' || cs[i - 1] == '+' || cs[i - 1] == '-')) {
                    nums.push(0);
                }
                while (!ops.isEmpty() && ops.peek() != '(') {
                    char pre = ops.peek();
                    //计算栈内的高优先级操作
                    if (map.get(pre) >= map.get(ch)) {
                        calc(nums, ops);
                    } else {
                        break;
                    }
                }
                ops.push(ch);
            }
        }
        while (!ops.isEmpty()) {
            calc(nums, ops);
        }
        return nums.peek();
    }

    private void calc(Stack<Integer> nums, Stack<Character> ops) {
        if (nums.isEmpty() || nums.size() < 2) return;
        if (ops.isEmpty()) return;

        int first = nums.pop();
        int second = nums.pop();
        char op = ops.pop();
        int ans = 0;
        if (op == '+') {
            ans = first + second;
        } else if (op == '-') {
            ans = second - first;
        } else if (op == '*') {
            ans = first * second;
        } else if (op == '/') {
            ans = second / first;
        }
        nums.push(ans);
    }


    private static int[] dfsForCalculate(String s, int start) {
        int n = s.length();
        int num = 0;
        char preSign = '+';
        int sum = 0;
        for (int i = start; i < n; i++) {
            char ch = s.charAt(i);
            if (ch == ' ') continue;
            boolean isDigit = Character.isDigit(ch);
            if (isDigit) {
                num = num * 10 + ch - '0';
                continue;
            }
            if (ch == '+' || ch == '-') {
                if (preSign == '+') {
                    sum += num;
                } else {
                    sum -= num;
                }
                preSign = ch;
                num = 0;
            } else if (ch == '(') {
                int[] res = dfsForCalculate(s, i + 1);
                num = res[0];
                i = res[1];
            } else if (ch == ')') {
                if (preSign == '+') {
                    sum += num;
                } else {
                    sum -= num;
                }
                return new int[]{sum, i};
            }
        }
        if (preSign == '+') {// 当前f由字符串到达结尾结束
            sum += num;// 结算结果
        } else {
            sum -= num;
        }
        return new int[]{sum, 0};
    }

    private static int dfsForCalculate(Deque<Character> queue) {
        Stack<Integer> stack = new Stack<>();
        char sign = '+';
        int num = 0;
        int res = 0;
        while (!queue.isEmpty()) {
            char c = queue.poll();
            if (Character.isDigit(c)) {
                num = num * 10 + c - '0';
            }
            if (c == '(') {
                num = dfsForCalculate(queue);
                //这里有一个坑，道理上遇不应该提前入栈，应该由下一轮的符号决定要不要入栈
                //是提前入栈数字之和，遇到下一次符号时上一次的符号是(，正好不需要对数字进行操作，导致坑又被填了，最好的办法是，当最后一个字符的时候，加入栈，否者等到后面一次
                if (!queue.isEmpty()) {
                    continue;
                }
            }
            //到达最后的时候要放入栈中，不然少数据，如果末尾是空格，触发后面一个条件，所以不要遇到空格就continue
            if (!Character.isDigit(c) && c != ' ' || queue.isEmpty()) {
                if (sign == '+') {
                    stack.push(num);
                } else if (sign == '-') {
                    stack.push(-num);
                } else if (sign == '*') {
                    stack.push(stack.pop() * num);
                } else if (sign == '/') {
                    stack.push(stack.pop() / num);
                }
                num = 0;
                sign = c;
            }

            if (c == ')') {
                break;
            }
        }

        for (int s : stack) {
            res += s;
        }
        return res;
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

    //[227].基本计算器
    public int calculate2(String s) {
        // 3+5 / 2
        //该题只有/ *优先级比较高，又没有括号，所以遇到符号的时候，更新根据前面的符号，对栈和当前值进行操作。
        s = s.trim();
        int n = s.length();
        char preSign = '+';
        int num = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < n; i++) {
            char ch = s.charAt(i);
            if (ch == ' ') continue;

            boolean isDig = Character.isDigit(ch);
            if (isDig) {
                num = num * 10 + ch - '0';
            }
            //如果第一个数负的，遇到-号，往栈中插入一个0，出栈不会影响总体值
            if (!isDig || i == n - 1) {
                if (preSign == '+') {
                    stack.push(num);
                } else if (preSign == '-') {
                    stack.push(-num);
                } else if (preSign == '/') {
                    stack.push(stack.pop() / num);
                } else if (preSign == '*') {
                    stack.push(stack.pop() * num);
                }
                //符号的时候更新数字
                preSign = ch;
                num = 0;
            }
        }

        int ans = 0;
        while (!stack.isEmpty()) {
            ans += stack.pop();
        }
        return ans;
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

        Stack<Integer> in;
        Stack<Integer> out;

        public MyQueue() {
            in = new Stack<>();
            out = new Stack<>();
        }

        public void push(int x) {
            in.push(x);
        }

        public int pop() {
            if (out.isEmpty()) {
                while (!in.isEmpty()) out.push(in.pop());
            }
            return out.pop();
        }

        public int peek() {
            if (out.isEmpty()) {
                while (!in.isEmpty()) out.push(in.pop());
            }
            return out.peek();
        }

        public boolean empty() {
            return out.isEmpty() && in.isEmpty();
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
    //[剑指 Offer 68 - I].二叉搜索树的最近公共祖先
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
//        int n = nums.length;
//        int[] left = new int[n + 1];
//        int[] right = new int[n + 1];
//        left[0] = 1;
//        right[n] = 1;
//        for (int i = 0; i < n; i++) {
//            left[i + 1] = left[i] * nums[i];
//        }
//        for (int i = n - 1; i >= 0; i--) {
//            right[i] = right[i + 1] * nums[i];
//        }
//
//        int[] res = new int[n];
//        for (int i = 0; i < n; i++) {
//            res[i] = left[i] * right[i + 1];
//        }
//        return res;

        int n = nums.length;
        int[] res = new int[n];
        int p = 1;
        for (int i = 0; i < n; i++) {
            res[i] = p;
            p *= nums[i];
        }
        int q = 1;
        for (int i = n - 1; i >= 0; i--) {
            res[i] *= q;
            q *= nums[i];
        }
        return res;
    }

    //[239].滑动窗口最大值
    public static int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        int[] ans = new int[n - k + 1];
        //单调队列
        LinkedList<Integer> queue = new LinkedList<>();
        int index = 0;
        for (int l = 0, r = 0; r < n; r++) {
            while (!queue.isEmpty() && nums[queue.peekLast()] < nums[r]) {
                queue.pollLast();
            }
            queue.offerLast(r);

            //满足k个就缩左边窗口
            if (r + 1 >= k) {
                int first = queue.peekFirst();
                ans[index++] = nums[first];
                l++;

                //更新之后发现最大值不在了，就删除掉
                if (first < l) {
                    queue.pollFirst();
                }
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

    //[242].有效的字母异位词
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) return false;
        int[] cnt = new int[26];
        for (int i = 0; i < s.length(); i++) {
            cnt[s.charAt(i) - 'a']++;
        }
        for (int i = 0; i < t.length(); i++) {
            cnt[t.charAt(i) - 'a']--;
        }
        for (int i = 0; i < 26; i++) {
            if (cnt[i] != 0) {
                return false;
            }
        }
        return true;
    }

    //[252].会议室
    public boolean canAttendMeetings(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        for (int i = 0; i < intervals.length; i++) {
            if (i + 1 < intervals.length && intervals[i][1] > intervals[i + 1][0]) {
                return false;
            }
        }
        return true;
    }

    //[253].会议室II
    public int minMeetingRooms(int[][] intervals) {
        //[[0, 30],[5, 10],[15, 20]]
        //思路，优先按照start排序，然后以end建立小根堆，如果当前的start >= 堆中最小的end，说明两个时间上面没有冲突，可以共用一个会议室。最终的堆大小就是最少会议室
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        PriorityQueue<Integer> queue = new PriorityQueue<>();

        for (int i = 0; i < intervals.length; i++) {
            int[] interval = intervals[i];
            if (!queue.isEmpty() && interval[0] >= queue.peek()) queue.poll();
            queue.offer(interval[1]);
        }
        return queue.size();
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

    //[273].整数转换英文表示
    public String numberToWords(int num) {
        String[] num2small = new String[]{"Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
        String[] num2Median = new String[]{"", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
        String[] num2Large = new String[]{"Billion", "Million", "Thousand", ""};
        if (num == 0) return num2small[0];

        StringBuilder sb = new StringBuilder();
        for (int i = (int) 1e9, j = 0; i >= 1; i /= 1000, j++) {
            if (num < i) continue;
            int a = num / i;
            String ans = "";
            if (a >= 100) {
                ans += num2small[a / 100] + " Hundred ";
                a %= 100;
            }
            if (a >= 20) {
                ans += num2Median[a / 10] + " ";
                a %= 10;
            }
            if (a != 0) {
                ans += num2small[a] + " ";
            }

            sb.append(ans).append(num2Large[j]).append(" ");
            num %= i;
        }
        while (sb.charAt(sb.length() - 1) == ' ') sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
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
        Arrays.fill(dp, 0x3f3f3f3f);
        dp[0] = 0;
        //物品是完全平方数
        for (int i = 1; i * i <= n; i++) {
            //背包是n
            for (int j = i * i; j <= n; i++) {
                dp[j] = Math.min(dp[j - i * i] + 1, dp[j]);
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

    //[295].数据流的中位数
    public class MedianFinder {

        PriorityQueue<Integer> small;
        PriorityQueue<Integer> large;

        public MedianFinder() {
            //这个是大根堆，保证堆顶是较大值
            small = new PriorityQueue<>((a, b) -> b - a);
            //这个是小根堆，保证堆顶是较小值
            large = new PriorityQueue<>((a, b) -> a - b);
        }

        public void addNum(int num) {
            if (small.size() > large.size()) {
                small.offer(num);
                large.offer(small.poll());
            } else {
                large.offer(num);
                small.offer(large.poll());
            }
        }

        public double findMedian() {
            if (small.size() == large.size()) {
                return (small.peek() + large.peek()) / 2.0d;
            } else if (small.size() > large.size()) {
                return small.peek();
            } else {
                return large.peek();
            }
        }
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
        //以i为结尾的最长递增子序列的长度
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
                //相等的时候，放在同一沓上，因为求得是递增子序列，得确保下一组一定是放不下再新建一组
                if (dp[mid] < num) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            //放最顶上去
            //maxLen，默认是左闭右开区间，left可以出界，就是新堆。
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
        //长度为i
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

    //[305].岛屿数量 II
    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        UnionFind3 uf = new UnionFind3(m * n);
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        //只有陆地才会被访问
        boolean[] visited = new boolean[m * n];
        List<Integer> res = new ArrayList<>();
        for (int[] position : positions) {
            int x = position[0];
            int y = position[1];
            int idx = x * n + y;
            if (!visited[idx]) {
                visited[idx] = true;
                //增加连通分量，刚开始为0
                uf.addCount();
                for (int[] direction : directions) {
                    int nx = x + direction[0];
                    int ny = y + direction[1];
                    int newIdx = nx * n + ny;
                    //被访问过就是陆地，没被访问过就不是陆地
                    if (nx < 0 || ny < 0 || nx >= m || ny >= n || !visited[newIdx] || uf.connected(idx, newIdx))
                        continue;
                    uf.union(idx, newIdx);
                }
            }
            res.add(uf.getCount());
        }
        return res;
    }

    public class UnionFind3 {
        int[] parent;
        int count;

        public UnionFind3(int size) {
            parent = new int[size];
            //每个海水连通分量都可以为自己
            for (int i = 0; i < size; i++) {
                parent[i] = i;
            }
            //这边因为陆地是动态添加的，所以初始化没有连通分量
            count = 0;
        }

        public int find(int p) {
            while (p != parent[p]) {
                parent[p] = parent[parent[p]];
                p = parent[p];
            }
            return p;
        }

        public void addCount() {
            count++;
        }

        public int getCount() {
            return count;
        }

        public boolean connected(int p, int q) {
            return find(p) == find(q);
        }

        public void union(int p, int q) {
            int rootP = find(p);
            int rootQ = find(q);
            parent[rootP] = rootQ;
            count--;
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

    //[307].区域和检索 - 数组可修改 （重点掌握）
    public class NumArray307 {

        private int[] tree;
        private int[] nums;

        private int lowbit(int x) {
            return x & (-x);
        }

        private int query(int x) {
            int ans = 0;
            for (int i = x; i > 0; i -= lowbit(i)) ans += tree[i];
            return ans;
        }

        private void add(int x, int u) {
            for (int i = x; i <= tree.length - 1; i += lowbit(i)) tree[i] += u;
        }

        public NumArray307(int[] nums) {
            int n = nums.length;
            this.nums = nums;
            this.tree = new int[n + 1];
            for (int i = 0; i < n; i++) add(i + 1, nums[i]);
        }

        public void update(int index, int val) {
            //此处额外注意，补的是差值
            add(index + 1, val - nums[index]);
            nums[index] = val;
        }

        public int sumRange(int left, int right) {
            return query(right + 1) - query(left);
        }
    }

    //[309].最佳买卖股票时机含冷冻期
    public int maxProfit4(int[] prices) {
        int n = prices.length;
        //冷冻时间为1天，买入 - 卖出 - 冷冻 - 买入
        int[][] dp = new int[n][2];
        if (n <= 1) return 0;

        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        dp[1][0] = Math.max(dp[0][0], dp[0][1] + prices[1]);
        dp[1][1] = Math.max(dp[0][1], -prices[1]);
        for (int i = 2; i < n; i++) {
            //没股票
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            //有股票，买入的状态，是在之前卖出的基础之上转移的
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 2][0] - prices[i]);
        }
        return dp[n - 1][0];
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

    //[311].稀疏矩阵的乘法
    public int[][] multiply(int[][] A, int[][] B) {
        List<int[]> aPoints = getNonZeroPoints(A);
        List<int[]> bPoints = getNonZeroPoints(B);

        int[][] res = new int[A.length][B[0].length];
        for (int[] a : aPoints) {
            for (int[] b : bPoints) {
                if (a[1] == b[0]) {
                    res[a[0]][b[1]] += A[a[0]][a[1]] * B[b[0]][b[1]];
                }
            }
        }
        return res;
    }

    private List<int[]> getNonZeroPoints(int[][] matrix) {
        List<int[]> points = new ArrayList<>();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] != 0) {
                    points.add(new int[]{i, j});
                }
            }
        }
        return points;
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

    //[322].零钱兑换
    public int coinChange(int[] coins, int amount) {
        int n = coins.length;
        int[] dp = new int[amount + 1];
        //dp[0] 金额为0， 需要的硬币数为0，求得是最小硬币数量，所以初始化为最大
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 0; i < n; i++) {
            for (int j = coins[i]; j <= amount; j++) {
                //要的是恰好零钱，如果dp[j - coins[i]]为最大值，意味着j - coins[i]金额没办法兑换。
                if (dp[j - coins[i]] != Integer.MAX_VALUE) {
                    dp[j] = Math.min(dp[j], dp[j - coins[i]] + 1);
                }
            }
        }
        //没办法兑换
        return dp[amount] == Integer.MAX_VALUE ? -1 : dp[amount];
    }

    //[324].摆动排序 II
    public void wiggleSort(int[] nums) {
        Arrays.sort(nums);
        //1| 1 2 2 | 3 3 4
        // 2 4 2 3 1 3 | 1
        int[] temp = nums.clone();
        int n = nums.length;
        for (int i = 0; i < n / 2; i++) {
            //较小值一半的位置倒序
            nums[2 * i] = temp[(n - 1) / 2 - i];
            //较大值倒序
            nums[2 * i + 1] = temp[n - 1 - i];
        }

        if ((n % 2) != 0) {
            nums[n - 1] = temp[0];
        }
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

    //[329].矩阵中的最长递增路径
    public int longestIncreasingPath(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[][] memo = new int[m][n];
        int ans = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                //当前最长的值
                if (memo[i][j] == 0) {
                    ans = Math.max(ans, dfsForLongestIncreasingPath(matrix, m, n, memo, i, j));
                }
            }
        }
        return ans;
    }

    private int dfsForLongestIncreasingPath(int[][] matrix, int m, int n, int[][] memo, int x, int y) {
        if (memo[x][y] != 0) return memo[x][y];

        int ans = 1;
        int[][] directs = new int[][]{{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
        for (int[] dir : directs) {
            int nx = x + dir[0];
            int ny = y + dir[1];
            if (nx < 0 || ny < 0 || nx >= m || ny >= n || matrix[nx][ny] <= matrix[x][y]) continue;
            ans = Math.max(ans, dfsForLongestIncreasingPath(matrix, m, n, memo, nx, ny) + 1);
        }

        memo[x][y] = ans;

        return ans;
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

    //[332].重新安排行程
    public List<String> findItinerary(List<List<String>> tickets) {
        LinkedList<String> res = new LinkedList<>();
        if (tickets == null || tickets.size() == 0) {
            return res;
        }
        Map<String, LinkedList<String>> graph = new HashMap<>();
        for (List<String> ticket : tickets) {
            graph.putIfAbsent(ticket.get(0), new LinkedList<>());
            graph.get(ticket.get(0)).add(ticket.get(1));
        }
        graph.values().forEach(x -> x.sort(String::compareTo));
        dfsForFindItinerary(graph, "JFK", res);
        return res;
    }

    private void dfsForFindItinerary(Map<String, LinkedList<String>> graph, String src, LinkedList<String> path) {
        LinkedList<String> items = graph.get(src);
        path.add(src);
        if (items == null) {
            return;
        }
        //做选择，这里就不要撤销了，因为存在唯一解，优先取字典序最小的就可以
        while (items.size() > 0) {
            String dest = items.removeFirst();
            dfsForFindItinerary(graph, dest, path);
        }
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
//        if (root == null) return 0;
//        if (map.containsKey(root)) return map.get(root);
//        int rob_it = root.val +
//                (root.left != null ? rob(root.left.left) + rob(root.left.right) : 0) +
//                (root.right != null ? rob(root.right.left) + rob(root.right.right) : 0);
//        int rob_not = rob(root.left) + rob(root.right);
//        int res = Math.max(rob_it, rob_not);
//        map.put(root, res);
//        return res;
        //前面一种方法，涉及到重复计算问题，当爷爷时，计算了儿子和孙子，当儿子时，又得计算孙子，所以有性能损耗。
        //后一种办法，只涉及到后序遍历思想，来更新父节点。
        int[] res = robAction(root);
        return Math.max(res[0], res[1]);
    }

    private int[] robAction(TreeNode root) {
        int[] res = new int[2];
        if (root == null) return res;

        int[] left = robAction(root.left);
        int[] right = robAction(root.right);

        //选择当前节点不偷，最大值取决于左右孩子最大值(可以偷，也可以不偷)
        res[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        //选择当前节点偷，最大值取决于左右孩子都不偷的情况
        res[1] = root.val + left[0] + right[0];
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

    //[341].扁平化嵌套列表迭代器
    interface NestedInteger {

        // @return true if this NestedInteger holds a single integer, rather than a nested list.
        public boolean isInteger();

        // @return the single integer that this NestedInteger holds, if it holds a single integer
        // Return null if this NestedInteger holds a nested list
        public Integer getInteger();

        // @return the nested list that this NestedInteger holds, if it holds a nested list
        // Return empty list if this NestedInteger holds a single integer
        public List<NestedInteger> getList();
    }

    public class NestedIterator implements Iterator<Integer> {
        // This is the interface that allows for creating nested lists.
        // You should not implement it, or speculate about its implementation

//        private Queue<Integer> queue = new LinkedList<>();
//        public NestedIterator(List<NestedInteger> nestedList) {
//            dfs(nestedList);
//        }
//
//        @Override
//        public Integer next() {
//            return hasNext() ? queue.poll() : -1;
//        }
//
//        @Override
//        public boolean hasNext() {
//            return !queue.isEmpty();
//        }
//
//        private void dfs(List<NestedInteger> nestedList) {
//            for(NestedInteger it : nestedList) {
//                if (it.isInteger()) {
//                    queue.offer(it.getInteger());
//                } else {
//                    dfs(it.getList());
//                }
//            }
//        }

        private Stack<NestedInteger> stack = new Stack<>();

        public NestedIterator(List<NestedInteger> nestedList) {
            //跟二叉树一样，前序遍历，是倒序进栈
            for (int i = nestedList.size() - 1; i >= 0; i--) {
                stack.push(nestedList.get(i));
            }
        }

        @Override
        public Integer next() {
            return hasNext() ? stack.pop().getInteger() : -1;
        }

        @Override
        public boolean hasNext() {
            if (stack.isEmpty()) return false;
            else {
                if (stack.peek().isInteger()) {
                    return true;
                } else {
                    List<NestedInteger> list = stack.pop().getList();
                    for (int i = list.size() - 1; i >= 0; i--) {
                        stack.push(list.get(i));
                    }
                    //可能有多个嵌套
                    return hasNext();
                }
            }
        }
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

    //[345].反转字符串中的元音字母
    public String reverseVowels(String s) {
        int n = s.length();
        int l = 0, r = n - 1;
        char[] str = s.toCharArray();
        while (l < r) {
            if (!isVowel(str[l])) {
                l++;
            } else if (!isVowel(str[r])) {
                r--;
            } else {
                char temp = str[l];
                str[l] = str[r];
                str[r] = temp;
                l++;
                r--;
            }
        }
        return String.valueOf(str);
    }

    private boolean isVowel(char ch) {
        return ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o';
    }

    //[347].前 K 个高频元素
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }

        int n = nums.length;
        List<Integer>[] bucket = new List[n + 1];
        for (int key : map.keySet()) {
            int i = map.get(key);
            if (bucket[i] == null) {
                bucket[i] = new ArrayList<>();
            }
            bucket[i].add(key);
        }

        int[] res = new int[k];
        for (int i = n, j = 0; i >= 1 && j < k; i--) {
            if (bucket[i] != null) {
                for (int w = 0; w < bucket[i].size() && j < k; w++) {
                    res[j++] = bucket[i].get(w);
                }
            }
        }
        return res;
    }

    //[349].两个数组的交集
    public int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> set1 = new HashSet<>();
        Set<Integer> resSet = new HashSet<>();
        for (int num : nums1) {
            set1.add(num);
        }
        for (int num : nums2) {
            if (set1.contains(num)) {
                resSet.add(num);
            }
        }
        int[] res = new int[resSet.size()];
        int index = 0;
        for (int num : resSet) {
            res[index++] = num;
        }
        return res;
    }

    //[354].俄罗斯套娃信封问题
    public int maxEnvelopes(int[][] envelopes) {
        int n = envelopes.length;
        //因为求的是递增子序列，逆序高度可以保证，相同宽度的情况下高度只能选择一个，
        //2,3
        //5,4
        //5,5
        //5,6
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

    //[355].设计推特
    public class Twitter {

        private class Tweet {
            int tweetId;
            //时间戳需要添加的，不然没办法按时间戳来排序
            long timestamp;
            Tweet next;

            public Tweet(int tweetId) {
                this.tweetId = tweetId;
                this.timestamp = System.currentTimeMillis();
            }
        }

        Map<Integer, Tweet> userTweet;
        Map<Integer, Set<Integer>> followers;

        public Twitter() {
            userTweet = new HashMap<>();
            followers = new HashMap<>();
        }

        public void postTweet(int userId, int tweetId) {
            Tweet cur = userTweet.get(userId);
            Tweet latest;
            if (cur == null) {
                latest = new Tweet(tweetId);
            } else {
                latest = new Tweet(tweetId);
                latest.next = cur;
            }
            userTweet.put(userId, latest);
        }

        public List<Integer> getNewsFeed(int userId) {
            Set<Integer> users = new HashSet<>();
            //把自己添加进去
            users.add(userId);
            Set<Integer> followee = followers.get(userId);
            if (followee != null) {
                users.addAll(followee);
            }

            PriorityQueue<Tweet> queue = new PriorityQueue<>((a, b) -> (int) (b.timestamp - a.timestamp));
            for (int user : users) {
                //没发表过，就不需要加
                if (!userTweet.containsKey(user)) {
                    continue;
                }
                queue.offer(userTweet.get(user));
            }

            List<Integer> res = new ArrayList<>();
            while (!queue.isEmpty()) {
                Tweet cur = queue.poll();
                res.add(cur.tweetId);
                if (res.size() == 10) {
                    return res;
                }

                if (cur.next != null) {
                    queue.offer(cur.next);
                }
            }
            return res;
        }

        public void follow(int followerId, int followeeId) {
            Set<Integer> followee = followers.getOrDefault(followerId, new HashSet<>());
            followee.add(followeeId);
            followers.put(followerId, followee);
        }

        public void unfollow(int followerId, int followeeId) {
            Set<Integer> followee = followers.get(followerId);
            if (followee != null) {
                followee.remove(followeeId);
            }
        }
    }

    //[357].计算各个位数不同的数字个数
    public int countNumbersWithUniqueDigits(int n) {
        //不要被题目标题诱惑，明显是0 ~ 10^n次方
        //n = 0，只能取0一个数
        if (n == 0) return 1;
        //1  10种不同的数字
        //2  9 * 9种不同的数字 + 1位数的不同数字
        //3  9 * 9 * 8种不同的数字 + 2位数的不同数字
        //11 9 * 9 * 8 * ... * 0种不同的数字 + 10位数的不同数字
        int res = 10, dp = 9;
        for (int i = 2; i <= Math.min(n, 10); i++) {
            dp = dp * (10 - i + 1);
            res += dp;
        }
        return res;
    }

    //[362].敲击计数器
    public class HitCounter {

        List<Integer> list;

        public HitCounter() {
            list = new ArrayList<>();
        }

        public void hit(int timestamp) {
            list.add(timestamp);
        }

        public int getHits(int timestamp) {
            int left = 0, right = list.size();
            while (left < right) {
                int mid = left + (right - left) / 2;
                if (list.get(mid) > timestamp) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            //右边界
            int r2 = left;

            left = -1;
            right = list.size() - 1;
            while (left < right) {
                int mid = left + (right - left + 1) / 2;
                if (list.get(mid) <= timestamp - 300) {
                    left = mid;
                } else {
                    right = mid - 1;
                }
            }
            //左边界
            int r1 = left + 1;
            return r2 - r1;
        }
    }

    //[368].最大整除子集
    public List<Integer> largestDivisibleSubset(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);

        int[] dp = new int[n];
        int max = 1, maxIndex = -1;
        for (int i = 0; i < n; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] % nums[j] == 0) {
                    dp[i] = Math.max(dp[j] + 1, dp[i]);
                }
            }
            if (dp[i] > max) {
                maxIndex = i;
                max = dp[i];
            }
        }
        List<Integer> res = new ArrayList<>();
        for (int i = maxIndex; i >= 0; i--) {
            if (nums[maxIndex] % nums[i] == 0 && dp[i] == max) {
                res.add(nums[i]);
                max--;
                maxIndex = i;
            }
        }
        return res;
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

    //[375].猜数字大小 II
    public int getMoneyAmount(int n) {
//        //区间dp问题， dp[1][1]，只有1数字的时候，最小的罚金
//        //从i到j的范围内确保胜利的最少金额，确保胜利就是最大罚金
//        //dp[i][j] = min i<=k<=j {k + max(dp[i][k-1], dp[k+1][j])}
//        //依赖下边和左边，所以倒序遍历
//        int[][] dp = new int[n + 1][n + 1];
//        //base case dp[i][i] = 0
//        for (int i = n - 1; i >= 1; i--) {
//            for (int j = i; j <= n; j++) {
//                if (j == i) {
//                    dp[i][j] = 0;
//                } else {
//                    //主要是外层取得是min操作，所以需要设置为最大值，而不是最小值，里面的max其实可以推倒出来，不是基于max值推倒出来的。
//                    dp[i][j] = Integer.MAX_VALUE;
//                    for (int k = i; k < j; k++) {
//                        dp[i][j] = Math.min(dp[i][j], Math.max(dp[i][k - 1], dp[k + 1][j]) + k);
//                    }
//                }
//            }
//        }
//        return dp[1][n];

        int[][] dp = new int[n + 1][n + 1];
        return dfsForGetMoneyAmount(dp, 1, n);
    }

    private int dfsForGetMoneyAmount(int[][] dp, int left, int right) {
        if (left >= right) return 0;

        if (dp[left][right] != 0) {
            return dp[left][right];
        }

        int res = Integer.MAX_VALUE;

        for (int i = left; i <= right; i++) {
            int cost = Math.max(dfsForGetMoneyAmount(dp, left, i - 1), dfsForGetMoneyAmount(dp, i + 1, right)) + i;
            res = Math.min(res, cost);
        }

        dp[left][right] = res;
        return res;
    }

    //[376].摆动序列
    public int wiggleMaxLength(int[] nums) {
        int n = nums.length;
        //长度1也被是被摆动序列，2的话需要完全不相等
        if (n < 2) return n;
        //i为结尾的上升子序列的最大长度
        int[] up = new int[n];
        //i为结尾的下降子序列的最大长度
        int[] down = new int[n];
        up[0] = 1;
        down[0] = 1;
        for (int i = 1; i < n; i++) {
            if (nums[i] == nums[i - 1]) {
                up[i] = up[i - 1];
                down[i] = down[i - 1];
            } else if (nums[i] > nums[i - 1]) {
                //严格上升，那么在下降子序列上+1
                up[i] = Math.max(down[i - 1] + 1, up[i - 1]);
                //严格上升，对下降子序列没有增益，所以保持之前的值
                down[i] = down[i - 1];
            } else {
                down[i] = Math.max(up[i - 1] + 1, down[i - 1]);
                up[i] = up[i - 1];
            }
        }
        return Math.max(up[n - 1], down[n - 1]);
    }

    //[377].组合总和 Ⅳ
    public int combinationSum4(int[] nums, int target) {
        int n = nums.length;
        int[] dp = new int[target + 1];
        dp[0] = 1;
        //实际求得是：排列数，所以容量在前，物品在后
        //如果是求组合数，那么物品在前，容量在后
        for (int j = 0; j <= target; j++) {
            for (int i = 0; i < n; i++) {
                if (j >= nums[i]) {
                    dp[j] += dp[j - nums[i]];
                }
            }
        }
        return dp[target];
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
    public class Solution384 {
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
            //请认真思考下，等概率是什么。
            //位置为0的概率为1/n, 位置为1的概率为n-1/n * 1/n-1，即第一次未被选中的概率 * 第二次选中的概率
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

    //[385].迷你语法分析器
    public NestedInteger deserialize(String s) {
//        if (s.charAt(0) != '[') return new NestedInteger(Integer.parseInt(s));
//        Stack<NestedInteger> stack = new Stack<>();
//        int num = 0, sign = 1;
//        for(int i = 0; i < s.length(); i++) {
//            char ch = s.charAt(i);
//            if (ch == '[') {
//                stack.push(new NestedInteger());
//            } else if (ch == '-' ) {
//                sign = -1;
//            } else if (ch == ',' || ch == ']') {
//                if (Character.isDigit(s.charAt(i-1))) {
//                    stack.peek().add(new NestedInteger(sign * num));
//                    sign = 1;
//                    num = 0;
//                }
//                //前面非数字的时候
//                if (ch == ']' && stack.size() > 1) {
//                    NestedInteger last = stack.pop();
//                    stack.peek().add(last);
//                }
//            } else {
//                num = num * 10 + (ch - '0');
//            }
//        }
//        return stack.peek();
        return null;
    }

    //[386].字典序排数
    public List<Integer> lexicalOrder(int n) {
//        List<Integer> res = new ArrayList<>();
//        //从1到9开始选择，刚开始1个数字，后序有个层的概念
//        for (int i = 1; i <= 9; i++) {
//            dfsForLexicalOrder(n, i, res);
//        }
//        return res;

        List<Integer> res = new ArrayList<>();
        int cur = 1;
        for (int i = 1; i <= n; i++) {
            res.add(cur);
            if (cur * 10 <= n) {
                cur *= 10;
            } else {
                while (cur % 10 == 9 || cur >= n) {
                    cur /= 10;
                }
                cur++;
            }
        }
        return res;
    }

    private void dfsForLexicalOrder(int n, int i, List<Integer> select) {
        if (i > n) {
            return;
        }
        select.add(i);
        //选择
        for (int j = 0; j <= 9; j++) {
            if (i * 10 + j > n) return;
            dfsForLexicalOrder(n, i * 10 + j, select);
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

    //[391].完美矩形
    public boolean isRectangleCover(int[][] rectangles) {
        int X1 = Integer.MAX_VALUE, Y1 = Integer.MAX_VALUE, X2 = Integer.MIN_VALUE, Y2 = Integer.MIN_VALUE;
        //奇数点最终为4个
        Set<String> oddPoints = new HashSet<>();
        int area = 0;
        for (int[] rectangle : rectangles) {
            int x1 = rectangle[0], y1 = rectangle[1];
            int x2 = rectangle[2], y2 = rectangle[3];
            List<String> points = new ArrayList<>();
            points.add(x1 + "," + y2);
            points.add(x1 + "," + y1);
            points.add(x2 + "," + y1);
            points.add(x2 + "," + y2);
            for (String point : points) {
                if (oddPoints.contains(point)) {
                    oddPoints.remove(point);
                } else {
                    oddPoints.add(point);
                }
            }

            X1 = Math.min(x1, X1);
            Y1 = Math.min(y1, Y1);
            X2 = Math.max(x2, X2);
            Y2 = Math.max(y2, Y2);
            area += Math.abs(x2 - x1) * Math.abs(y2 - y1);
        }
        if (oddPoints.size() != 4) return false;
        if (Math.abs(X2 - X1) * Math.abs(Y2 - Y1) != area) return false;
        if (!oddPoints.contains(X1 + "," + Y2)) return false;
        if (!oddPoints.contains(X1 + "," + Y1)) return false;
        if (!oddPoints.contains(X2 + "," + Y2)) return false;
        if (!oddPoints.contains(X2 + "," + Y1)) return false;
        return true;
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

    //[393].UTF-8 编码验证
    public static boolean validUtf8(int[] data) {
        //识别第一个是多少个连续的1，后面遇到10就数量--，遇到数量>0且不是10的return false
        //   0000 0000-0000 007F | 0xxxxxxx
        //   0000 0080-0000 07FF | 110xxxxx 10xxxxxx
        //   0000 0800-0000 FFFF | 1110xxxx 10xxxxxx 10xxxxxx
        //   0001 0000-0010 FFFF | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        //最低8位来搞数据
        int count = 0;
        for (int item : data) {
            if (count > 0) {
                if (item >> 6 != 0b10) {
                    return false;
                }
                count--;
            } else if (item >> 7 == 0) {
                count = 0;
            } else if (item >> 5 == 0b110) {
                count = 1;
            } else if (item >> 4 == 0b1110) {
                count = 2;
            } else if (item >> 3 == 0b11110) {
                count = 3;
            } else {
                //首位不是110,1110,11110开头的，是非法的
                return false;
            }
        }
        return count == 0;
    }

    //[394].字符串解码
    public String decodeString(String s) {
//        return dfsForDecodeString(s, 0)[1];

        Stack<Integer> numbers = new Stack<>();
        Stack<String> strings = new Stack<>();
        char[] arr = s.toCharArray();
        int num = 0;
        StringBuilder last = new StringBuilder();
        for (char ch : arr) {
            if (ch >= '0' && ch <= '9') {
                num = num * 10 + (ch - '0');
            } else if (ch == '[') {
                //将数字和字符都放入栈中
                numbers.push(num);
                //[之前的字符串，有可能是个空串
                strings.push(last.toString());
                last = new StringBuilder();
                num = 0;
            } else if (ch == ']') {
                //数字栈出栈 * last
                int cnt = numbers.pop();
                String preStr = strings.pop();
                StringBuilder temp = new StringBuilder(preStr);
                while (cnt-- > 0) {
                    temp.append(last);
                }
                last = temp;
            } else {
                last.append(ch);
            }
        }
        return last.toString();
    }

    public String[] dfsForDecodeString(String s, int start) {
        //遇到[开始递归，遇到]就返回到上一层
        StringBuilder res = new StringBuilder();
        int num = 0;
        for (int i = start; i < s.length(); i++) {
            if (s.charAt(i) >= '0' && s.charAt(i) <= '9') {
                num = num * 10 + s.charAt(i) - '0';
            } else if (s.charAt(i) == '[') {
                String[] tmp = dfsForDecodeString(s, i + 1);
                i = Integer.parseInt(tmp[0]);
                while (num > 0) {
                    res.append(tmp[1]);
                    num--;
                }
            } else if (s.charAt(i) == ']') {
                return new String[]{String.valueOf(i), res.toString()};
            } else {
                res.append(s.charAt(i));
            }
        }
        return new String[]{String.valueOf(s.length()), res.toString()};
    }

    //[395].至少有 K 个重复字符的最长子串
    public int longestSubstring(String s, int k) {
        int n = s.length();
        //不限定字符串种类
        //右端点往右，如果字符出现过，字符串数量>k满足条件; 没出现过，字符串数量不满足条件

        int ans = 0;
        //通过限定字符串种类数量来划分
        //右端点往右，字符串种类增加
        //左端点往右，字符串种类减少
        int[] cnt = new int[26];
        for (int p = 1; p <= 26; p++) {
            //每次更新计算
            Arrays.fill(cnt, 0);
            //tot为种类， sum为都>k的字符个数
            for (int i = 0, j = 0, tot = 0, sum = 0; j < n; j++) {
                int u = s.charAt(j) - 'a';
                cnt[u]++;
                // 如果添加到 cnt 之后为 1，说明字符总数 +1
                if (cnt[u] == 1) tot++;
                // 如果添加到 cnt 之后等于 k，说明该字符从不达标变为达标，达标数量 + 1
                if (cnt[u] == k) sum++;

                while (tot > p) {
                    int t = s.charAt(i) - 'a';
                    cnt[t]--;
                    // 如果添加到 cnt 之后为 0，说明字符总数 -1
                    if (cnt[t] == 0) tot--;
                    // 如果添加到 cnt 之后等于 k - 1，说明该字符从达标变为不达标，达标数量 - 1
                    if (cnt[t] == k - 1) sum--;
                    i++;
                }
                if (sum == tot) {
                    ans = Math.max(ans, j - i + 1);
                }
            }
        }
        return ans;
    }

    //[396].旋转函数
    public int maxRotateFunction(int[] nums) {
        //F(n) - F(n-1) = sum - size * nums[size - n];
        int n = nums.length;
        int[] dp = new int[n];
        int sum = 0;
        for (int i = 0; i < n; i++) {
            dp[0] += i * nums[i];
            sum += nums[i];
        }

        int max = Integer.MIN_VALUE;
        for (int i = 1; i < n; i++) {
            dp[i] = dp[i - 1] + sum - n * nums[n - i];
            max = Math.max(max, dp[i]);
        }
        return max;
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
        Random random;

        public Solution398(int[] nums) {
            this.nums = nums;
            this.random = new Random();
        }

        public int pick(int target) {
//            List<Integer> res = new ArrayList<>();
//            for (int i = 0; i < nums.length; i++) {
//                if (target == nums[i]) {
//                    res.add(i);
//                }
//            }
//            Random random = new Random();
//            return res.get(random.nextInt(res.size()));
            int ans = -1;
            for (int i = 0, cnt = 0; i < nums.length; i++) {
                if (target == nums[i]) {
                    cnt++;
                    if (random.nextInt(cnt) == 0) {
                        ans = i;
                    }
                }
            }
            return ans;
        }
    }

    //[399].除法求值
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        int n = equations.size();
        int index = 0;

        //两两不相等，最多有2n个节点，并查集走的是index，需要转化成index
        UnionFind2 uf = new UnionFind2(n * 2);
        Map<String, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < n; i++) {
            List<String> equation = equations.get(i);

            String p = equation.get(0);
            String q = equation.get(1);
            if (!indexMap.containsKey(p)) {
                indexMap.put(p, index++);
            }
            if (!indexMap.containsKey(q)) {
                indexMap.put(q, index++);
            }
            uf.union(indexMap.get(p), indexMap.get(q), values[i]);
        }

        double[] res = new double[queries.size()];
        for (int i = 0; i < queries.size(); i++) {
            List<String> query = queries.get(i);
            String p = query.get(0);
            String q = query.get(1);
            if (!indexMap.containsKey(p) || !indexMap.containsKey(q)) {
                res[i] = -1.0d;
            } else {
                res[i] = uf.connected(indexMap.get(p), indexMap.get(q));
            }
        }
        return res;
    }

    private class UnionFind2 {
        int[] parent;
        double[] weight;

        public UnionFind2(int n) {
            parent = new int[n];
            weight = new double[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
                weight[i] = 1.0d;
            }
        }

        public int find(int x) {
            if (x != parent[x]) {
                int origin = parent[x];
                //路径压缩，它的老子更新 为 它老子节点的根节点
                parent[x] = find(parent[x]);
                //路径压缩完之后，将权重更新，其实应该是从上到下更新
                weight[x] *= weight[origin];
            }
            return parent[x];
        }

        public void union(int p, int q, double v) {
            int rootP = find(p);
            int rootQ = find(q);
            if (rootP == rootQ) return;

            //分子连上分母
            parent[rootP] = rootQ;
            //    rootX  --?---> rootY
            //     /            /
            //    /w[x]        / w[y]
            //   x -----v---> y
            //w[x] * w[rootX] = v * w[y]
            //目的是查找的时的路径压缩，能够正常换算p节点和rootP的权值，p会直接接入到rootP上
            weight[rootP] = v * weight[q] / weight[p];
        }

        public double connected(int p, int q) {
            int rootP = find(p);
            int rootQ = find(q);
            if (rootP != rootQ) return -1.0d;
            //因为find，带路径压缩，p会直接接入到rootP上，q会直接接入到rootQ上，所以可以直接相除得到值
            return weight[p] / weight[q];
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

    //[401].二进制手表
    public static List<String> readBinaryWatch(int turnedOn) {
        List<String> res = new ArrayList<>();
        for (int i = 0; i < 1024; i++) {
            int hours = i >> 6;
            int minutes = i & 63;
            if (hours < 12 && minutes < 60 && Integer.bitCount(i) == turnedOn) {
                res.add(hours + ":" + (minutes < 10 ? "0" + minutes : minutes));
            }
        }
        return res;
    }

    //[402].移掉 K 位数字
    public static String removeKdigits(String num, int k) {
        if (num.equals("0")) return "0";
        //"1432219", k = 3  => 1219
        // 123456, k = 3 =>
        //维护单调递增栈，后面发现前面的大，就把栈中的移除，保证后面是单调递增
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < num.length(); i++) {
            while (!stack.isEmpty() && num.charAt(stack.peek()) > num.charAt(i) && k > 0) {
                stack.pop();
                k--;
            }
            //避免前导0
            if (stack.isEmpty() && num.charAt(i) == '0') continue;
            stack.push(i);
        }

        while (k-- > 0) stack.pop();
        if (stack.isEmpty()) {
            return "0";
        }
        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            sb.insert(0, num.charAt(stack.pop()));
        }
        return sb.toString();
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

    //[407].接雨水 II
    public int trapRainWater(int[][] heightMap) {
        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> a[2] - b[2]);
        int m = heightMap.length, n = heightMap[0].length;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < n; i++) {
            queue.offer(new int[]{0, i, heightMap[0][i]});
            queue.offer(new int[]{m - 1, i, heightMap[m - 1][i]});
            visited[0][i] = visited[m - 1][i] = true;
        }

        for (int i = 0; i < m; i++) {
            queue.offer(new int[]{i, 0, heightMap[i][0]});
            queue.offer(new int[]{i, n - 1, heightMap[i][n - 1]});
            visited[i][0] = visited[i][n - 1] = true;
        }

        int ans = 0;
        int[][] directs = new int[][]{{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0], y = cur[1], h = cur[2];
            for (int[] direct : directs) {
                int nx = x + direct[0];
                int ny = y + direct[1];
                if (nx < 0 || nx >= m || ny < 0 || ny >= n || visited[nx][ny]) continue;
                if (h > heightMap[nx][ny]) {
                    ans += h - heightMap[nx][ny];
                }
                queue.offer(new int[]{nx, ny, Math.max(heightMap[nx][ny], h)});
                visited[nx][ny] = true;
            }
        }
        return ans;
    }

    //[413].等差数列划分
    public int numberOfArithmeticSlices(int[] nums) {
        int n = nums.length;
        //[0~i]之间新增的连续子数组的个数
        int[] dp = new int[n];

        int sum = 0;
        for (int i = 2; i < n; i++) {
            //相当于找到一个等差元素，增量了以i为结尾的所有等差数列，而且还额外增加了一个 i-2, i-1, i的数列
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) {
                dp[i] = dp[i - 1] + 1;
            }
            sum += dp[i];
        }
        return sum;
    }

    //[415].字符串相加
    public String addStrings(String num1, String num2) {
        int carry = 0;
        int i = num1.length() - 1, j = num2.length() - 1;
        StringBuilder sb = new StringBuilder();
        while (i >= 0 || j >= 0 || carry != 0) {
            int n1 = i >= 0 ? num1.charAt(i) - '0' : 0;
            int n2 = j >= 0 ? num2.charAt(j) - '0' : 0;
            int sum = n1 + n2 + carry;
            sb.insert(0, sum % 10);
            carry = sum / 10;
            i--;
            j--;
        }
        return sb.toString();
    }

    //[416].分割等和子集
    public boolean canPartition(int[] nums) {
        int n = nums.length;
        int sum = 0;
        for (int num : nums) sum += num;
        if (sum % 2 != 0) return false;
        sum /= 2;
        //0到i的和是否凑成j
        boolean[][] dp = new boolean[n][sum + 1];
        for (int i = 0; i < n; i++) dp[i][0] = true;

        //依赖上和左，所以正序遍历，而且取值又是[n-1][sum]
        for (int i = 1; i < n; i++) {
            for (int j = 1; j <= sum; j++) {
                if (j < nums[i]) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    //放得下，可以选择，也可以不选择
                    //选择i之后能凑成j的 | 不选择i，前面i-1个是否能凑成j
                    dp[i][j] = dp[i][j - nums[i]] | dp[i - 1][j];
                }
            }
        }
        return dp[n - 1][sum];
    }

    //[417].太平洋大西洋水流问题
    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        List<List<Integer>> res = new ArrayList<>();
        int m = heights.length, n = heights[0].length;
        //能够流向太平洋的点
        boolean[][] res1 = new boolean[m][n];
        //能够流向大西洋的点
        boolean[][] res2 = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            if (!res1[i][0]) {
                dfsForPacificAtlantic(heights, i, 0, res1);
            }
            if (!res2[i][n - 1]) {
                dfsForPacificAtlantic(heights, i, n - 1, res2);
            }
        }
        for (int j = 0; j < n; j++) {
            if (!res1[0][j]) {
                dfsForPacificAtlantic(heights, 0, j, res1);
            }
            if (!res2[m - 1][j]) {
                dfsForPacificAtlantic(heights, m - 1, j, res2);
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (res1[i][j] && res2[i][j]) {
                    res.add(Arrays.asList(i, j));
                }
            }
        }
        return res;
    }

    int[][] directs = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    private void dfsForPacificAtlantic(int[][] heights, int x, int y, boolean[][] res) {
        res[x][y] = true;
        for (int[] direct : directs) {
            int nx = x + direct[0];
            int ny = y + direct[1];
            if (nx < 0 || ny < 0 || nx >= heights.length || ny >= heights[0].length) continue;
            //后面的只能比前面的大于或者等于
            if (res[nx][ny] || heights[x][y] > heights[nx][ny]) continue;
            dfsForPacificAtlantic(heights, nx, ny, res);
        }
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

    //[421].数组中两个数的最大异或值
    public class Solution421 {

        class XorTrie {
            XorTrie[] children = new XorTrie[2];
        }

        XorTrie root = new XorTrie();

        private void build(int[] nums) {
            for (int num : nums) {
                XorTrie cur = root;
                for (int i = 30; i >= 0; i--) {
                    int bit = num >> i & 1;
                    if (cur.children[bit] == null) {
                        cur.children[bit] = new XorTrie();
                    }
                    cur = cur.children[bit];
                }
            }
        }

        private int search(int num) {
            XorTrie cur = root;
            int res = 0;
            for (int i = 30; i >= 0; i--) {
                int bit = num >> i & 1;
                int xor = bit ^ 1;

                if (cur.children[xor] != null) {
                    cur = cur.children[xor];
                    //相反方向有值，意味着异或之后肯定是1
                    res |= 1 << i;
                } else {
                    cur = cur.children[bit];
                }
            }
            return res;
        }

        public int findMaximumXOR(int[] nums) {
            if (nums.length == 0) return 0;
            build(nums);
            int max = nums[0];
            for (int i = 1; i < nums.length; i++) {
                max = Math.max(max, search(nums[i]));
            }
            return max;
        }
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

    //[424].替换后的最长重复字符
    public int characterReplacement(String s, int k) {
//        int[] cnt = new int[26];
//        int maxCount = 0;
//        int l = 0;
//        for (int r = 0; r < s.length(); ) {
//            cnt[s.charAt(r) - 'A']++;
//            maxCount = Math.max(maxCount, cnt[s.charAt(r) - 'A']);
//            r++;
//
//            //最大的窗口 > 某个字符最大的数量+ 其他字符替换的k次，就需要缩窗口
//            if (r - l > maxCount + k) {
//                cnt[s.charAt(l) - 'A']--;
//                l++;
//            }
//        }
//        //左窗口的位置就是最大的重复串位置
//        return s.length() - l;

        //宫水三叶解法
        int ans = 0;
        int[] cnt = new int[26];
        int n = s.length();
        for (int l = 0, r = 0; r < n; r++) {
            int cur = s.charAt(r) - 'A';
            cnt[cur]++;
            //想想什么时候需要缩窗口，肯定是总数量 - 数量最多的字符 > k 的时候
            while (!check(cnt, k)) {
                cnt[s.charAt(l) - 'A']--;
                l++;
            }

            ans = Math.max(ans, r - l + 1);
        }
        return ans;
    }

    private boolean check(int[] cnt, int k) {
        int maxCount = 0, sum = 0;
        for (int i = 0; i < 26; i++) {
            maxCount = Math.max(maxCount, cnt[i]);
            sum += cnt[i];
        }
        return sum - maxCount <= k;
    }

    //[427].建立四叉树
    public class Solution427 {
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

        public Node construct(int[][] grid) {
            int n = grid.length;
            int[][] preSum = new int[n + 1][n + 1];
            for (int i = 1; i <= n; i++) {
                for (int j = 1; j <= n; j++) {
                    preSum[i][j] = preSum[i - 1][j] + preSum[i][j - 1] - preSum[i - 1][j - 1] + grid[i - 1][j - 1];
                }
            }

            return dfs(0, 0, n, n, preSum);
        }

        private Node dfs(int r0, int c0, int r1, int c1, int[][] preSum) {
            int sum = preSum[r1][c1] - preSum[r0][c1] - preSum[r1][c0] + preSum[r0][c0];
            if (sum == 0) {
                return new Node(false, true, null, null, null, null);
            } else if (sum == (r1 - r0) * (c1 - c0)) {
                return new Node(true, true, null, null, null, null);
            }
            Node node = new Node(false, false);
            node.topLeft = dfs(r0, c0, (r0 + r1) / 2, (c0 + c1) / 2, preSum);
            node.topRight = dfs(r0, (c0 + c1) / 2, (r0 + r1) / 2, c1, preSum);
            node.bottomLeft = dfs((r0 + r1) / 2, c0, r1, (c0 + c1) / 2, preSum);
            node.bottomRight = dfs((r0 + r1) / 2, (c0 + c1) / 2, r1, c1, preSum);
            return node;
        }
    }

    //[428].序列化和反序列化 N 叉树
    public class Solution428 {
        public String serialize(Node root) {
            if (root == null) return "";
            StringBuilder sb = new StringBuilder();
            dfsForSerialize(root, sb);
            return sb.toString();
        }

        private void dfsForSerialize(Node root, StringBuilder sb) {
            sb.append(root.val).append("_");
            sb.append(root.children.size()).append("_");

            for (Node child : root.children) {
                dfsForSerialize(child, sb);
            }
        }

        public Node deserialize(String data) {
            if (data == null || data.length() == 0) return null;
            String[] list = data.split("_");
            LinkedList<String> items = new LinkedList<>();
            for (String item : list) {
                items.add(item);
            }
            return dfsForDeserialize(items);
        }

        private Node dfsForDeserialize(LinkedList<String> items) {
            if (items == null || items.isEmpty()) return null;
            int val = Integer.parseInt(items.pollFirst());
            int size = Integer.parseInt(items.pollFirst());

            Node root = new Node(val);
            root.children = new ArrayList<>(size);

            for (int i = 0; i < size; i++) {
                root.children.add(dfsForDeserialize(items));
            }
            return root;
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

    //[430].扁平化多级双向链表
    public class Solution430 {
        class Node {
            public int val;
            public Node prev;
            public Node next;
            public Node child;
        }

        public Node flatten(Node head) {
            dfsForFlatten(head);
            return head;
        }

        //返回尾节点
        public Node dfsForFlatten(Node head) {
            Node cur = head;
            Node last = null;
            while (cur != null) {
                Node next = cur.next;
                if (cur.child != null) {
                    Node childLast = dfsForFlatten(cur.child);

                    //处理child节点
                    cur.next = cur.child;
                    cur.child.prev = cur;
                    cur.child = null;

                    //处理next不为空
                    if (next != null) {
                        next.prev = childLast;
                        childLast.next = next;
                    }
                    last = childLast;
                } else {
                    last = cur;
                }
                cur = next;
            }
            return last;
        }
    }

    //[432].全 O(1) 的数据结构
    public class AllOne {

        class Node {
            int count;
            Set<String> keys = new HashSet<>();
            Node pre;
            Node next;

            public Node(int count) {
                this.count = count;
            }
        }

        Node head;
        Node tail;
        Map<String, Node> map;

        public AllOne() {
            head = new Node(-1000);
            tail = new Node(-1000);
            head.next = tail;
            tail.pre = head;
            map = new HashMap<>();
        }

        public void inc(String key) {
            if (map.containsKey(key)) {
                Node node = map.get(key);
                node.keys.remove(key);
                Node next = null;
                if (node.next.count != node.count + 1) {
                    next = new Node(node.count + 1);
                    node.next.pre = next;
                    next.next = node.next;
                    next.pre = node;
                    node.next = next;
                } else {
                    next = node.next;
                }
                next.keys.add(key);
                map.put(key, next);
                removeNode(node);
            } else {
                Node node = null;
                if (head.next.count == 1) {
                    node = head.next;
                } else {
                    node = new Node(1);
                    head.next.pre = node;
                    node.next = head.next;
                    node.pre = head;
                    head.next = node;
                }
                node.keys.add(key);
                map.put(key, node);
            }
        }

        public void dec(String key) {
            Node node = map.get(key);
            node.keys.remove(key);
            if (node.count == 1) {
                map.remove(key);
            } else {
                Node pre = null;
                if (node.pre.count != node.count - 1) {
                    pre = new Node(node.count - 1);
                    node.pre.next = pre;
                    pre.pre = node.pre;
                    pre.next = node;
                    node.pre = pre;
                } else {
                    pre = node.pre;
                }
                pre.keys.add(key);
                map.put(key, pre);
            }
            removeNode(node);
        }

        public String getMaxKey() {
            Node pre = tail.pre;
            for (String str : pre.keys) return str;
            return "";
        }

        public String getMinKey() {
            Node next = head.next;
            for (String str : next.keys) return str;
            return "";
        }

        private void removeNode(Node node) {
            if (node.keys.size() == 0) {
                node.pre.next = node.next;
                node.next.pre = node.pre;
            }
        }
    }

    //[433].最小基因变化
    public int minMutation(String start, String end, String[] bank) {
        AtomicInteger res = new AtomicInteger(Integer.MAX_VALUE);
        dfsForMinMutation(start, end, bank, new HashSet<>(), res);
        return res.get() == Integer.MAX_VALUE ? -1 : res.get();
    }

    private void dfsForMinMutation(String current, String end, String[] bank, Set<String> select, AtomicInteger minCount) {
        if (current.equals(end)) {
            minCount.set(Math.min(minCount.get(), select.size()));
            return;
        }
        //选择只能从bank里找
        for (String str : bank) {
            //select用来加速，防止重复计算
            if (!checkNeighbor(current, str) || select.contains(str)) {
                continue;
            }
            select.add(str);
            dfsForMinMutation(str, end, bank, select, minCount);
            select.remove(str);
        }
    }

    private boolean checkNeighbor(String first, String second) {
        int n = first.length(), diff = 0;
        for (int i = 0; i < n; i++) {
            diff += first.charAt(i) == second.charAt(i) ? 0 : 1;
        }
        return diff == 1;
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
            //刚开始窗口不相等的时候，词频肯定不相等
            if (diffA == diffP) res.add(l);
        }
        return res;
    }

    //[440].字典序的第K小数字
    public int findKthNumber(int n, int k) {
        int ans = 1;
        while (k > 1) {
            int cnt = getCnt(ans, n);
            if (cnt < k) {
                //从兄弟节点找
                ans++;
                k -= cnt;
            } else {
                //就在ans为前缀的节点中找，只不过是从下一层节点找
                ans *= 10;
                k--;
            }
        }
        return ans;
    }

    public int getCnt(int x, int limit) {
        String a = String.valueOf(x), b = String.valueOf(limit);
        int m = a.length(), n = b.length(), k = n - m;
        int ans = 0, u = Integer.parseInt(b.substring(0, m));
        //以x为前缀，<=limit的个数， 位数少于limit的位数其实是合法值
        //x = 12  limit = 1234  k = 2
        // 12 1个
        // 120 ... 129  10个
        // 1200 ... 1234  35个
        for (int i = 0; i < k; i++) ans += Math.pow(10, i);

        if (x == u) {
            //x = 12 limit = 123， 120...123共三个
            ans += limit - x * Math.pow(10, k) + 1;
        } else if (x < u) {
            //x = 11 limit = 123   110 111 ... 119共十个
            ans += Math.pow(10, k);
        }
        return ans;
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

    //[447].回旋镖的数量
    public int numberOfBoomerangs(int[][] points) {
        int n = points.length;
        int res = 0;
        for (int i = 0; i < n; i++) {
            Map<Integer, Integer> distanceCount = new HashMap<>();
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    int distance = (points[i][0] - points[j][0]) * (points[i][0] - points[j][0])
                            + (points[i][1] - points[j][1]) * (points[i][1] - points[j][1]);
                    distanceCount.put(distance, distanceCount.getOrDefault(distance, 0) + 1);
                }
            }
            for (int count : distanceCount.values()) {
                if (count > 1) {
                    res += count * (count - 1);
                }
            }
        }
        return res;
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

    //[451].根据字符出现频率排序
    public static String frequencySort(String s) {
        //桶排序，时间复杂度更低
        Map<Character, Integer> map = new HashMap<>();
        int max = 0;
        for (char ch : s.toCharArray()) {
            map.put(ch, map.getOrDefault(ch, 0) + 1);
            max = Math.max(map.get(ch), max);
        }
        StringBuffer[] sbs = new StringBuffer[max + 1];
        for (int i = 0; i <= max; i++) {
            sbs[i] = new StringBuffer();
        }
        for (Map.Entry<Character, Integer> entry : map.entrySet()) {
            sbs[entry.getValue()].append(entry.getKey());
        }

        StringBuffer res = new StringBuffer();
        for (int i = max; i > 0; i--) {
            StringBuffer sb = sbs[i];
            for (int j = 0; j < sb.length(); j++) {
                for (int k = 0; k < i; k++) {
                    res.append(sb.charAt(j));
                }
            }
        }
        return res.toString();
    }

    //[452].用最少数量的箭引爆气球
    public int findMinArrowShots(int[][] points) {
        Arrays.sort(points, (a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
        int e = points[0][1];
        int ans = 1;
        for (int i = 1; i < points.length; i++) {
            int[] cur = points[i];
            if (cur[0] > e) {
                //没交集，新的一根针
                ans++;
                e = cur[1];
            }
        }
        return ans;
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

    //[457].环形数组是否存在循环
    public boolean circularArrayLoop(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            //虚拟节点，所以fast可以设置在+1的位置上
            int slow = i, fast = getNext(nums, i);
            while (nums[slow] * nums[fast] > 0 && nums[fast] * nums[getNext(nums, fast)] > 0) {
                if (slow == fast) {
                    if (slow != getNext(nums, slow)) {
                        return true;
                    } else {
                        break;
                    }
                }
                slow = getNext(nums, slow);
                fast = getNext(nums, getNext(nums, fast));
            }

            //无法构成环, 把遍历过的元素都标记为0(题目不可能存在0)
            int x = i;
            while (nums[x] * nums[getNext(nums, x)] > 0) {
                int tmp = getNext(nums, x);
                nums[x] = 0;
                x = tmp;
            }
        }
        return false;
    }

    private int getNext(int[] nums, int i) {
        int n = nums.length;
        // 正值：(i + nums[i]) % n
        // 负值：(i + nums[i]) % n + n
        return ((nums[i] + i) % n + n) % n;
    }

    //[460].LFU 缓存
    public class LFUCache {
        Map<Integer, Integer> keyToVal;
        Map<Integer, Integer> keyToFreq;
        Map<Integer, LinkedHashSet<Integer>> freqToKeys;
        int capacity;
        int minFreq;

        public LFUCache(int capacity) {
            keyToVal = new HashMap<>();
            keyToFreq = new HashMap<>();
            freqToKeys = new HashMap<>();
            this.capacity = capacity;
        }

        public int get(int key) {
            if (!keyToVal.containsKey(key)) return -1;
            updateFreq(key);
            return keyToVal.get(key);
        }

        public void put(int key, int value) {
            if (capacity <= 0) return;
            if (keyToVal.containsKey(key)) {
                keyToVal.put(key, value);
                updateFreq(key);
            } else {
                if (capacity <= keyToVal.size()) {
                    LinkedHashSet<Integer> keys = freqToKeys.get(minFreq);
                    int removeKey = keys.iterator().next();
                    keyToVal.remove(removeKey);
                    keyToFreq.remove(removeKey);
                    keys.remove(removeKey);
                    if (keys.size() == 0) {
                        freqToKeys.remove(minFreq);
                    }
                }

                keyToVal.put(key, value);
                keyToFreq.put(key, 1);
                freqToKeys.putIfAbsent(1, new LinkedHashSet<>());
                freqToKeys.get(1).add(key);
                minFreq = 1;
            }
        }

        private void updateFreq(int key) {
            int freq = keyToFreq.get(key);
            int newFreq = freq + 1;
            freqToKeys.putIfAbsent(newFreq, new LinkedHashSet<>());
            LinkedHashSet<Integer> newKeys = freqToKeys.get(newFreq);
            newKeys.add(key);

            LinkedHashSet<Integer> keys = freqToKeys.get(freq);
            keys.remove(key);
            if (keys.size() == 0) {
                freqToKeys.remove(freq);
                if (freq == minFreq) {
                    minFreq++;
                }
            }
        }
    }

    public class LFUCache2 {
        Map<Integer, Node> keyMap;
        Map<Integer, DoublyLinkedList> freqMap;
        int capacity;
        int minFreq;

        public LFUCache2(int capacity) {
            keyMap = new HashMap<>();
            freqMap = new HashMap<>();
            this.capacity = capacity;
        }

        public int get(int key) {
            Node node = keyMap.get(key);
            if (node == null) return -1;
            updateFreq(node);
            return node.value;
        }

        public void put(int key, int value) {
            if (capacity == 0) return;
            Node node = keyMap.get(key);
            if (node != null) {
                node.value = value;
                updateFreq(node);
            } else {
                if (keyMap.size() == capacity) {
                    DoublyLinkedList linkedList = freqMap.get(minFreq);
                    Node deleteNode = linkedList.tail.pre;

                    linkedList.removeNode(deleteNode);
                    keyMap.remove(deleteNode.key);
                }

                node = new Node(key, value);
                keyMap.put(key, node);

                freqMap.put(1, new DoublyLinkedList());
                DoublyLinkedList linkedList = freqMap.get(1);
                linkedList.addNode2Head(node);

                minFreq = 1;
            }
        }

        private void updateFreq(Node node) {
            int freq = node.freq;
            DoublyLinkedList linkedList = freqMap.get(freq);
            linkedList.removeNode(node);

            if (freq == minFreq && linkedList.head.next == linkedList.tail) {
                minFreq++;
            }

            node.freq++;

            freqMap.putIfAbsent(freq + 1, new DoublyLinkedList());
            linkedList = freqMap.get(freq + 1);
            linkedList.addNode2Head(node);
        }


        class Node {
            int key;
            int value;
            int freq = 1;
            Node pre;
            Node next;

            public Node(int key, int value) {
                this.key = key;
                this.value = value;
            }
        }

        class DoublyLinkedList {
            Node head;
            Node tail;

            public DoublyLinkedList() {
                head = new Node(-1, -1);
                tail = new Node(-1, -1);
                head.next = tail;
                tail.pre = head;
            }

            public void removeNode(Node node) {
                node.pre.next = node.next;
                node.next.pre = node.pre;
                node.next = null;
                node.pre = null;
            }

            public void addNode2Head(Node node) {
                node.next = head.next;
                node.next.pre = node;
                node.pre = head;
                head.next = node;
            }
        }
    }

    //[462].最少移动次数使数组元素相等 II
    public int minMoves2(int[] nums) {
        //中位数，就是最少移动次数，常规操作肯定是排序之后，取中间的
        int n = nums.length;
        //用快速选择比较好，平均O(n)， 最坏情况O(n2)
        int mid = select(nums, 0, n - 1, n / 2);
        int sum = 0;
        for (int num : nums) {
            sum += Math.abs(mid - num);
        }
        return sum;
    }

    private int partition(int[] nums, int left, int right) {
        int pivot = nums[right];
        int i = left;
        for (int j = left; j <= right; j++) {
            if (nums[j] < pivot) {
                swap(nums, i, j);
                i++;
            }
        }
        swap(nums, i, right);
        return i;
    }

    private int select(int[] nums, int left, int right, int k) {
        if (left == right) return nums[left];
        int index = partition(nums, left, right);
        if (k == index) return nums[k];
        else if (k < index) return select(nums, left, index - 1, k);
        else return select(nums, index + 1, right, k);
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

    //[464].我能赢吗
    public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        //所有的和都比期望值小，先手不会赢，所以返回false
        if (maxChoosableInteger * (maxChoosableInteger + 1) / 2 < desiredTotal) {
            return false;
        }
        Map<Integer, Boolean> memo = new HashMap<>();
        return dfs(maxChoosableInteger, desiredTotal, 0, 0, memo);
    }

    private boolean dfs(int maxChoosableInteger, int desiredTotal, int state, int total, Map<Integer, Boolean> memo) {
        if (!memo.containsKey(state)) {
            boolean res = false;
            for (int i = 0; i < maxChoosableInteger; i++) {
                if (((state >> i) & 1) == 1) {
                    continue;
                }

                //当前已经超过了期望值，先手赢得比赛
                if (total + i + 1 >= desiredTotal) {
                    res = true;
                    break;
                }
                //后手输了比赛
                if (!dfs(maxChoosableInteger, desiredTotal, state | (1 << i), total + i + 1, memo)) {
                    res = true;
                    break;
                }
            }
            memo.put(state, res);
        }
        return memo.get(state);
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

    //[473].火柴拼正方形
    public boolean makesquare(int[] matchsticks) {
        if (null == matchsticks || matchsticks.length < 4) {
            return false;
        }
        int n = matchsticks.length;
        int sum = 0;
        for (int matchstick : matchsticks) {
            sum += matchstick;
        }
        if (sum % 4 != 0) return false;
        return dfsForMakesquare(matchsticks, 0, 0, 0, 0, 0, sum / 4);
    }

    private boolean dfsForMakesquare(int[] matchsticks, int index, int a, int b, int c, int d, int side) {
        if (index == matchsticks.length) {
            return a == b && b == c && c == d && d == a;
        }
        if (a > side || b > side || c > side || d > side) {
            return false;
        }
        int stick = matchsticks[index];
        return dfsForMakesquare(matchsticks, index + 1, a + stick, b, c, d, side)
                || dfsForMakesquare(matchsticks, index + 1, a, b + stick, c, d, side)
                || dfsForMakesquare(matchsticks, index + 1, a, b, c + stick, d, side)
                || dfsForMakesquare(matchsticks, index + 1, a, b, c, d + stick, side);
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

    //[477].汉明距离总和
    public int totalHammingDistance(int[] nums) {
        int ans = 0, n = nums.length;
        //原本要O(N^2) -> O(32N)，单独从每一位看待，距离总和=1的个数*0的个数
        for (int i = 0; i < 32; i++) {
            int ones = 0;
            for (int num : nums) {
                if (((num >> i) & 1) == 1) {
                    ones++;
                }
            }
            ans += ones * (n - ones);
        }
        return ans;
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

    //[479].最大回文数乘积
    public int largestPalindrome(int n) {
        if (n == 1) return 9;
        //利用贪心的策略，优先取99 * xx ==> 回文
        //其实可以通过计算出一个回文值，然后再去判断这个回文值能不能被99以下的数整除，如果能，就返回该回文。
        int upper = (int) Math.pow(10, n) - 1;
        for (int i = upper; i >= 0; i--) {
            long p = i, t = i;
            //根据左边的数，计算回文数
            while (t != 0) {
                p = p * 10 + (t % 10);
                t /= 10;
            }

            //判断能否被整除
            for (long j = upper; j * j >= p; j--) {
                if (p % j == 0) return (int) (p % 1337);
            }
        }
        return -1;
    }

    //[480].滑动窗口中位数
    public class Solution480 {

        PriorityQueue<Integer> small;
        PriorityQueue<Integer> large;

        public double[] medianSlidingWindow(int[] nums, int k) {
            small = new PriorityQueue<>((a, b) -> b - a);
            large = new PriorityQueue<>();
            int n = nums.length;
            double[] res = new double[n - k + 1];
            for (int l = 0, r = 0; r < n; r++) {
                add(nums[r]);
                if (r - l + 1 >= k) {
                    res[r + 1 - k] = median();
                    remove(nums[l]);
                    l++;
                }
            }
            return res;
        }

        private void remove(int num) {
            if (small.contains(num)) {
                small.remove(num);
                //移除完，防止原来small小于large，平衡下
                if (small.size() < large.size()) {
                    small.offer(large.poll());
                }
            } else {
                large.remove(num);
                if (large.size() < small.size()) {
                    large.offer(small.poll());
                }
            }
        }

        private void add(int num) {
            if (small.size() > large.size()) {
                small.offer(num);
                large.offer(small.poll());
            } else {
                large.offer(num);
                small.offer(large.poll());
            }
        }

        private double median() {
            if (small.size() == large.size()) {
                return (small.peek() + large.peek()) / 2.0d;
            } else if (small.size() > large.size()) {
                return small.peek();
            } else {
                return large.peek();
            }
        }
    }

    //[481].神奇字符串
    public static int magicalString(int n) {
        StringBuilder sb = new StringBuilder("122");
        boolean one = true;
        for (int i = 2; sb.length() < n; i++) {
            int num = sb.charAt(i) - '0';
            if (one) {
                while (num-- > 0) {
                    sb.append(1);
                }
                one = false;
            } else {
                while (num-- > 0) {
                    sb.append(2);
                }
                one = true;
            }
        }
        int res = 0;
        for (int i = 0; i < n; i++) {
            if (sb.charAt(i) == '1') res++;
        }
        return res;
    }

    //[486].预测赢家
    public static boolean PredictTheWinner(int[] nums) {
        int n = nums.length;
        //dp[i][j] 从i到j当前玩家最大分差值
        int[][] dp = new int[n][n];
        //base case： 当为一个元素的时候，先手获得分差值为nums[i]
        for (int i = 0; i < n; i++) dp[i][i] = nums[i];

        //依赖下和左，所以第一维度倒序遍历
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                //最大分差值
                dp[i][j] = Math.max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1]);
            }
        }
        //最大分差值是否大于0
        return dp[0][n - 1] >= 0;
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

    //[494].目标和
    public int findTargetSumWays(int[] nums, int target) {
        //x - (sum - x) = target => x = (target + sum) /2;
        //背包容量为x
        int sum = 0, n = nums.length;
        for (int num : nums) sum += num;
        //奇数，凑不了target
        if ((target + sum) % 2 != 0) return 0;
        int size = (target + sum) / 2;

        if (size < 0) size = -size;
        int[] dp = new int[size + 1];
        //求得是组合数，所以这边初始值为1
        //如果是求最值，这边是0，其他是边界值
        dp[0] = 1;
        for (int i = 0; i < n; i++) {//容量
            for (int j = size; j >= nums[i]; j--) {//背包，01，降维之后，需要考虑降序遍历
                dp[j] += dp[j - nums[i]];
            }
        }
        return dp[size];
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

    //[497].非重叠矩形中的随机点
    public class Solution497 {

        int[][] rects;
        TreeMap<Integer, int[]> map;
        int total;
        Random random;

        public Solution497(int[][] rects) {
            this.rects = rects;
            for (int[] rect : rects) {
                total += (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1);
                map.put(total, rect);
            }
            this.random = new Random();
        }

        public int[] pick() {
            int randomArea = random.nextInt(total);
            int ceilingArea = map.ceilingKey(randomArea);
            //被选中的区间
            int[] rect = map.get(ceilingArea);
            int width = rect[2] - rect[0] + 1;
            //多出来的空间，那么需要映射到区间中
            int offset = ceilingArea - randomArea;
            return new int[]{rect[0] + offset % width, rect[1] + offset / width};
        }
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

    //[508].出现次数最多的子树元素和
    public int[] findFrequentTreeSum(TreeNode root) {
        AtomicInteger maxCount = new AtomicInteger(0);
        Map<Integer, Integer> count = new HashMap<>();
        dfsForFindFrequentTreeSum(root, count, maxCount);

        List<Integer> res = new ArrayList<>();
        for (int num : count.keySet()) {
            if (count.get(num).equals(maxCount.get())) {
                res.add(num);
            }
        }
        int[] result = new int[res.size()];
        for (int i = 0; i < res.size(); i++) {
            result[i] = res.get(i);
        }
        return result;
    }

    private int dfsForFindFrequentTreeSum(TreeNode root, Map<Integer, Integer> count, AtomicInteger max) {
        if (root == null) return 0;
        int left = dfsForFindFrequentTreeSum(root.left, count, max);
        int right = dfsForFindFrequentTreeSum(root.right, count, max);
        int sum = left + right + root.val;
        count.put(sum, count.getOrDefault(sum, 0) + 1);
        max.set(Math.max(max.get(), count.get(sum)));
        return sum;
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

    //[补充题12].二叉树的下一个节点
    //[510].二叉搜索树中的中序后继 II
    public Node inorderSuccessor(Node x) {
        if (x.right != null) {
            x = x.right;
            while (x.left != null) x = x.left;
            return x;
        } else {
            //x.parent.right == x 效率更高，实际效率100%
            while (x.parent != null && x.parent.left != x) {
                x = x.parent;
            }
            return x.parent;
        }
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

    //[516].最长回文子序列
    public static int longestPalindromeSubseq(String s) {
        int n = s.length();
        //i到j之间的回文子序列的最长长度
        int[][] dp = new int[n][n];
        for (int i = 0; i < n; i++) {
            dp[i][i] = 1;
        }
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    //这边利用了为0的情况，如果bb相等，那么这边就是dp[1][0]+2 = 2
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }

    //[518].零钱兑换 II
    public int change(int amount, int[] coins) {
        int n = coins.length;
        int[] dp = new int[amount + 1];
        //需要的是组合数，当为0的时候，为1种组合，因为后面的状态转移是+，所以需要设置成1
        dp[0] = 1;
        for (int i = 0; i < n; i++) { //物品维度
            for (int j = coins[i]; j <= amount; j++) { //容量维度
                dp[j] += dp[j - coins[i]];
            }
        }
        //没法兑换的组合数为0
        return dp[amount];
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

    //[524].通过删除字母匹配到字典里最长单词
    public String findLongestWord(String s, List<String> dictionary) {
        //通过删除s的字符来匹配字典中的单词，那么相对顺序不会发生变化，意思就是求字典中的单词是s的子序列，而且是自然序最小的最长字符串
        //长度降序，长度相等自然序增序
        Collections.sort(dictionary, (a, b) -> a.length() == b.length() ? a.compareTo(b) : b.length() - a.length());
        for (String dic : dictionary) {
            int i = 0;
            for (int j = 0; i < dic.length() && j < s.length(); j++) {
                if (dic.charAt(i) == s.charAt(j)) {
                    i++;
                }
            }
            if (i == dic.length()) {
                return dic;
            }
        }
        return "";
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

    //[529].扫雷游戏
    public char[][] updateBoard(char[][] board, int[] click) {
        int[][] direct = new int[][]{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
        int x = click[0], y = click[1];
        if (board[x][y] == 'M') {
            board[x][y] = 'X';
        } else {
            dfsForUpdateBoard(board, x, y, direct);
        }
        return board;
    }

    private void dfsForUpdateBoard(char[][] board, int x, int y, int[][] direct) {
        int m = board.length, n = board[0].length;
        //先查看附近周围有没有隐藏炸弹，并计数
        int bomb = 0;
        for (int[] d : direct) {
            int newX = x + d[0], newY = y + d[1];
            if (newX < 0 || newY < 0 || newX >= m || newY >= n) {
                continue;
            }
            if (board[newX][newY] == 'M') {
                bomb++;
            }
        }
        board[x][y] = bomb > 0 ? (char) (bomb + '0') : 'B';
        if (board[x][y] == 'B') {
            for (int[] d : direct) {
                int newX = x + d[0], newY = y + d[1];
                //只能把E的消失掉
                if (newX < 0 || newY < 0 || newX >= m || newY >= n || board[newX][newY] != 'E') {
                    continue;
                }
                dfsForUpdateBoard(board, newX, newY, direct);
            }
        }
    }

    //[532].数组中的 k-diff 数对
    public int findPairs(int[] nums, int k) {
        Map<Integer, Integer> count = new HashMap<>();
        for (int num : nums) {
            count.put(num, count.getOrDefault(num, 0) + 1);
        }
        int ans = 0;
        for (int key : count.keySet()) {
            if (k == 0) {
                if (count.get(key) > 1) {
                    ans++;
                }
            } else {
                if (count.containsKey(key + k)) {
                    ans++;
                }
            }
        }
        return ans;
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
    public static int singleNonDuplicate(int[] nums) {
//        int n = nums.length;
//        int l = 0, r = n - 1;
//        while (l < r) {
//            int mid = l + (r - l) / 2;
//            boolean halfOdd = ((r - mid) % 2 == 0);
//            if (nums[mid - 1] == nums[mid]) {
//                if (halfOdd) {
//                    r = mid - 2;
//                } else {
//                    l = mid + 1;
//                }
//            } else if (nums[mid + 1] == nums[mid]) {
//                if (halfOdd) {
//                    l = mid + 2;
//                } else {
//                    r = mid - 1;
//                }
//            } else {
//                return nums[mid];
//            }
//        }
//        return nums[l];

        int n = nums.length;
        int l = 0, r = n - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] == nums[mid ^ 1]) {
                l = mid + 1;
            } else {
                r = mid;
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

    //[543].二叉树的直径
    public int diameterOfBinaryTree(TreeNode root) {
        //题目求得是任意节点之间的最大距离，很有可能是出现在子树中
        AtomicInteger max = new AtomicInteger(1);
        dfsForDiameterOfBinaryTree(root, max);
        return max.get() - 1;
    }

    private int dfsForDiameterOfBinaryTree(TreeNode root, AtomicInteger max) {
        if (root == null) return 0;

        int left = dfsForDiameterOfBinaryTree(root.left, max);
        int right = dfsForDiameterOfBinaryTree(root.right, max);
        max.set(Math.max(max.get(), left + right + 1));
        return Math.max(left, right) + 1;
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

    //[553].最优除法
    public String optimalDivision(int[] nums) {
        //要使得值最大，那么分母必须最小，因为是整数，所以后面的数连除是最小
        int n = nums.length;
        if (n == 0) return "";
        if (n == 1) return "" + nums[0];
        if (n == 2) return nums[0] + "/" + nums[1];
        StringBuilder sb = new StringBuilder(nums[0] + "/(" + nums[1]);
        for (int i = 2; i < n; i++) {
            sb.append("/" + nums[i]);
        }
        sb.append(")");
        return sb.toString();
    }

    //[554].砖墙
    public int leastBricks(List<List<Integer>> wall) {
        Map<Integer, Integer> countMap = new HashMap<>();
        int n = wall.size();
        int max = 0;
        for (List<Integer> row : wall) {
            int edge = 0;
            for (int i = 0; i < row.size() - 1; i++) {
                edge += row.get(i);
                countMap.put(edge, countMap.getOrDefault(edge, 0) + 1);
                max = Math.max(max, countMap.get(edge));
            }
        }
        return n - max;
    }

    //[556].下一个更大元素 III
    public int nextGreaterElement(int n) {
//        char[] a = String.valueOf(n).toCharArray();
//        int len = a.length;
//        //找到倒数第一个递减区间的最小值
//        int l = len - 2;
//        while (l >= 0 && a[l] >= a[l + 1]) l--;
//        if (l < 0) return -1;
//
//        int r = len - 1;
//        while (r >= 0 && a[l] >= a[r]) r--;
//        char temp = a[l];
//        a[l] = a[r];
//        a[r] = temp;
//        reverse(a, l + 1, len - 1);
//        return Integer.parseInt(new String(a));
        //1234764
        char[] a = String.valueOf(n).toCharArray();
        Stack<Integer> stack = new Stack<>();
        int len = a.length, leftIndex = -1, rightIndex = -1;
        for (int i = len - 1; i >= 0; i--) {
            //维护一个单调递增栈，遇到小的，压掉它，直到压不掉为止，最后一次出栈的必定是第一个大于的最小值
            while (!stack.isEmpty() && a[i] < a[stack.peek()]) {
                leftIndex = i;
                rightIndex = stack.pop();
            }

            if (leftIndex != -1) {
                break;
            }
            stack.push(i);
        }

        if (leftIndex == -1) return -1;
        char t = a[leftIndex];
        a[leftIndex] = a[rightIndex];
        a[rightIndex] = t;

        reverse(a, leftIndex + 1, len - 1);
        return Integer.parseInt(new String(a));
    }

    private void reverse(char[] a, int left, int right) {
        while (left < right) {
            char t = a[left];
            a[left] = a[right];
            a[right] = t;
            left++;
            right--;
        }
    }

    //[557].反转字符串中的单词 III
    public String reverseWords3(String s) {
        char[] arr = s.toCharArray();
        int left = 0;
        for (int i = 0; i <= s.length(); i++) {
            if (i == s.length() || s.charAt(i) == ' ') {
                int right = i - 1;
                while (left < right) {
                    char temp = arr[right];
                    arr[right] = arr[left];
                    arr[left] = temp;
                    left++;
                    right--;
                }
                left = i + 1;
            }
        }
        return String.valueOf(arr);
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

    //[564].寻找最近的回文数
    public String nearestPalindromic(String n) {
        int len = n.length();
        Set<Long> set = new HashSet<>();
        //通过取前半段数字分别-1, +0, +1来枚举最小值，其中-1,+1,会导致位数发生变化，实际就是左右边界值也是需要考虑的。
        //99 => 101
        //100 => 99
        set.add((long) Math.pow(10, len - 1) - 1);
        set.add((long) Math.pow(10, len) + 1);
        String leftSub = n.substring(0, (len + 1) / 2);
        int left = Integer.parseInt(leftSub);
        long self = Long.parseLong(n);
        for (int i = left - 1; i <= left + 1; i++) {
            StringBuilder sb = new StringBuilder();
            sb.append(i);
            String reverse = sb.reverse().toString();
            long num = Long.parseLong(i + reverse.substring(len & 1));
            set.add(num);
        }

        long ans = -1;
        for (long candidate : set) {
            if (candidate == self) continue;
            if (ans == -1
                    || Math.abs(ans - self) > Math.abs(candidate - self)
                    || (Math.abs(ans - self) == Math.abs(candidate - self) && candidate < ans)) {
                ans = candidate;
            }
        }
        return String.valueOf(ans);
    }

    //[565].数组嵌套
    public int arrayNesting(int[] nums) {
        int res = 0;
        //0~n-1的特性说明了，可能会有多个分组，相同的分组元素可以有环，不同的分组不会有环
        //一个分组遍历完了之后，不会影响下一个组
        Set<Integer> visited = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            int k = i, count = 0;
            while (!visited.contains(nums[k])) {
                visited.add(nums[k]);
                k = nums[k];
                count++;
            }
            //跳出循环有重复的，更新最大值，如果之前被遍历过，count就会为0，不会更新结果
            res = Math.max(res, count);
        }
        return res;
    }

    //[567].字符串的排列
    public boolean checkInclusion(String s1, String s2) {
//        int m = s1.length(), n = s2.length();
//        if (m > n) return false;
//        int[] cnt = new int[26];
//        //逆向思维，窗口内如果大于0，说明需要缩窗口，否则如果right - left + 1 == m
//        for (int i = 0; i < m; i++) {
//            --cnt[s1.charAt(i) - 'a'];
//        }
//
//        for (int right = 0, left = 0; right < n; right++) {
//            int ch = s2.charAt(right) - 'a';
//            ++cnt[ch];
//
//            while (cnt[ch] > 0) {
//                --cnt[s2.charAt(left) - 'a'];
//                left++;
//            }
//            if (right - left + 1 == m) {
//                return true;
//            }
//        }
//        return false;

        //宫水三叶的解法，非常易懂
        int m = s1.length(), n = s2.length();
        if (m > n) return false;
        int[] cnt1 = new int[26], cnt2 = new int[26];
        for (int i = 0; i < m; i++) cnt1[s1.charAt(i) - 'a']++;
        for (int i = 0; i < m; i++) cnt2[s2.charAt(i) - 'a']++;
        if (check(cnt1, cnt2)) return true;
        for (int i = m; i < n; i++) {
            //右边增加一个值
            cnt2[s2.charAt(i) - 'a']++;
            //左边减少一个值
            cnt2[s2.charAt(i - m) - 'a']--;
            //判断是否相等
            if (check(cnt1, cnt2)) {
                return true;
            }
        }
        return false;
    }

    private boolean check(int[] cnt1, int[] cnt2) {
        for (int i = 0; i < 26; i++) {
            if (cnt1[i] != cnt2[i]) return false;
        }
        return true;
    }

    //[572].另一棵树的子树
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        if (root == null) return false;
        return isSamTree(root, subRoot) || isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot);
    }

    private boolean isSamTree(TreeNode root, TreeNode subRoot) {
        if (root == null && subRoot == null) return true;
        if (root == null || subRoot == null) return false;
        if (root.val != subRoot.val) return false;
        return isSamTree(root.left, subRoot.left) && isSamTree(root.right, subRoot.right);
    }

    //[575].分糖果
    public int distributeCandies(int[] candyType) {
        Set<Integer> set = new HashSet<>();
        for (int candy : candyType) {
            set.add(candy);
        }
        return Math.min(candyType.length / 2, set.size());
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

    //[587].安装栅栏
    int[] subtraction(int[] a, int[] b) { // 向量相减
        return new int[]{a[0] - b[0], a[1] - b[1]};
    }

    double cross(int[] a, int[] b) { // 叉乘
        return a[0] * b[1] - a[1] * b[0];
    }

    double getArea(int[] a, int[] b, int[] c) { // 向量 ab 转为 向量 ac 过程中扫过的面积
        return cross(subtraction(b, a), subtraction(c, a));
    }

    public int[][] outerTrees(int[][] trees) {
        Arrays.sort(trees, (a, b) -> {
            return a[0] != b[0] ? a[0] - b[0] : a[1] - b[1];
        });
        int n = trees.length, tp = 0;
        int[] stk = new int[n + 10];
        boolean[] vis = new boolean[n + 10];
        stk[++tp] = 0; // 不标记起点
        for (int i = 1; i < n; i++) {
            int[] c = trees[i];
            while (tp >= 2) {
                int[] a = trees[stk[tp - 1]], b = trees[stk[tp]];
                if (getArea(a, b, c) < 0) vis[stk[tp--]] = false;
                else break;
            }
            stk[++tp] = i;
            vis[i] = true;
        }
        int size = tp;
        for (int i = n - 1; i >= 0; i--) {
            if (vis[i]) continue;
            int[] c = trees[i];
            while (tp > size) {
                int[] a = trees[stk[tp - 1]], b = trees[stk[tp]];
                if (getArea(a, b, c) < 0) tp--;
                    // vis[stk[tp--]] = false; // 非必须
                else break;
            }
            stk[++tp] = i;
            // vis[i] = true; // 非必须
        }
        int[][] ans = new int[tp - 1][2];
        for (int i = 1; i < tp; i++) ans[i - 1] = trees[stk[i]];
        return ans;
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

    //[599].两个列表的最小索引总和
    public static String[] findRestaurant(String[] list1, String[] list2) {
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < list1.length; i++) {
            map.put(list1[i], i);
        }
        int min = Integer.MAX_VALUE;
        List<String> res = new ArrayList<>();
        for (int i = 0; i < list2.length; i++) {
            String item = list2[i];
            if (map.containsKey(item)) {
                int j = map.get(item);
                if (i + j < min) {
                    res.clear();
                    res.add(item);
                    min = i + j;
                } else if (i + j == min) {
                    res.add(item);
                }
            }
        }
        return res.toArray(new String[res.size()]);
    }

    //[606].根据二叉树创建字符串
    public String tree2str(TreeNode root) {
        if (root == null) return "";
        StringBuilder sb = new StringBuilder();
        dfs(root, sb);
        return sb.substring(1, sb.length() - 1);
    }

    private void dfs(TreeNode root, StringBuilder sb) {
        sb.append("(");
        sb.append(root.val);
        if (root.left != null) dfs(root.left, sb);
        else if (root.right != null) sb.append("()");

        if (root.right != null) dfs(root.right, sb);
        sb.append(")");
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

    //[617].合并二叉树
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) return null;
        if (root1 != null && root2 == null) {
            return root1;
        } else if (root2 != null && root1 == null) {
            return root2;
        } else {
            TreeNode root = new TreeNode(root1.val + root2.val);
            root.left = mergeTrees(root1.left, root2.left);
            root.right = mergeTrees(root1.right, root2.right);
            return root;
        }
    }

    //[621].任务调度器
    public int leastInterval(char[] tasks, int n) {
        //举个例子：AAAABB 2 -> AB_AB_A__A前半部分只取决于n和最大的个数，后半部分取决于剩余的个数
        //前半部分 = (max-1) * (n+1) 后半部分 = 最大的数量的个数
        int[] cnt = new int[26];
        int max = 0;
        for (char ch : tasks) {
            cnt[ch - 'A']++;
            max = Math.max(max, cnt[ch - 'A']);
        }
        int sameCount = 0;
        for (int count : cnt) {
            if (max == count) {
                sameCount++;
            }
        }
        //当n==0的时候，公式不正确
        int res = (max - 1) * (n + 1) + sameCount;
        return Math.max(res, tasks.length);
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

    //[628].三个数的最大乘积
    public int maximumProduct(int[] nums) {
        int n = nums.length;
        if (n < 3) return 0;

        Arrays.sort(nums);
        int a = nums[n - 1] * nums[n - 2] * nums[n - 3];
        //有可能负数，并且只有两两是负数的时候才可能最大
        int b = nums[0] * nums[1] * nums[n - 1];
        return Math.max(a, b);
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

    //[636].函数的独占时间
    public static int[] exclusiveTime(int n, List<String> logs) {
        //start end start end
        //start start end end
        //因为有嵌套，两个start差值，两个end之间的差值，start和end之间的差值都需要更新，所以定义一个pre
        //end时，出战，更新pre
        //start时，进站，更新pre，如果栈不空，说明有嵌套，需要更新前面的时间。否则进站。
        Stack<Integer> stack = new Stack<>();
        int[] res = new int[n];
        String[] first = logs.get(0).split(":");

        stack.push(Integer.parseInt(first[0]));
        int pre = Integer.parseInt(first[2]);
        for (int i = 1; i < logs.size(); i++) {
            String[] split = logs.get(i).split(":");
            if (split[1].equals("start")) {
                //更新上一个节点的时间
                if (!stack.isEmpty()) {
                    res[stack.peek()] += Integer.parseInt(split[2]) - pre;
                }
                //更新start时间
                pre = Integer.parseInt(split[2]);
                //等待配对，进栈
                stack.push(Integer.parseInt(split[0]));
            } else {
                res[stack.peek()] += Integer.parseInt(split[2]) - pre + 1;
                //更新start时间
                pre = Integer.parseInt(split[2]) + 1;
                //配对成功，出栈
                stack.pop();
            }
        }
        return res;
    }

    //[637].二叉树的层平均值
    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> res = new ArrayList<>();
        if (root == null) return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            long sum = 0;
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();
                sum += cur.val;

                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }
            }
            res.add(sum * 1.0d / size);
        }
        return res;
    }

    //[640].求解方程
    public String solveEquation(String equation) {
        if (equation == null || equation.length() == 0) return "No solution";
        //"2x=x"
        //大概的想法，把等式替换成+-，按照=分割，然后按照+分割，然后求出x的系数，合并成如下式子
        //Ax + y = Bx + z， （A-B）x = z -y， 如果A-B = 0 且 z-y = 0，然后无限。如果 A-B = 0且z-y != 0 ，无解。否则返回解
        equation = equation.replaceAll("-", "+-");
        String left = equation.split("=")[0];
        String right = equation.split("=")[1];

        int A = 0, y = 0, B = 0, z = 0;
        for (String s : left.split("\\+")) {
            if (s.contains("x")) {
                String replace = s.replace("x", "");
                A += replace.length() == 0 ? 1 : replace.equals("-") ? -1 : Integer.parseInt(replace);
            } else if (s.length() > 0) {
                y += Integer.parseInt(s);
            }
        }
        for (String s : right.split("\\+")) {
            if (s.contains("x")) {
                String replace = s.replace("x", "");
                A += replace.length() == 0 ? 1 : replace.equals("-") ? -1 : Integer.parseInt(replace);
            } else if (s.length() > 0) {
                z += Integer.parseInt(s);
            }
        }

        if (A == B && y == z) return "Infinite solutions";
        else if (A == B && y != z) return "No solution";
        else return "x=" + (z - y) / (A - B);
    }

    //[653].两数之和 IV - 输入 BST
    public boolean findTarget(TreeNode root, int k) {
        if (root == null) return false;
        Set<Integer> set = new HashSet<>();
        return dfs(root, k, set);
    }

    private boolean dfs(TreeNode root, int k, Set<Integer> set) {
        if (root == null) return false;
        int val = root.val;
        if (set.contains(k - val)) {
            return true;
        }
        set.add(val);

        return dfs(root.left, k, set) || dfs(root.right, k, set);
    }

    //[658].找到 K 个最接近的元素
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        int n = arr.length;
//        int idx = findUpperRight(arr, 0, n - 1, x);
//        int left = idx - 1, right = idx;
//        List<Integer> res = new ArrayList<>();
//        while(k > 0) {
//            if(isLeftClose(arr, left, right, x)) {
//                res.add(arr[left]);
//                left--;
//            } else {
//                res.add(arr[right]);
//                right++;
//            }
//            k--;
//        }
//        Collections.sort(res);
//        return res;

        //left找的是最优左边界， right最多能到n-k的位置
        //画图理解:
        //x    |mid      mid+k|    r = mid, 排除掉右边
        //|mid     mid+k|     x    l = mid+1， 排除掉左边一个
        //|mid   x        mid+k|   r = mid, 排除掉右边
        //|mid        x   mid+k|   l = mid+1, 排除掉左边一个
        int left = 0, right = n - k;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (x - arr[mid] > arr[mid + k] - x) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        List<Integer> res = new ArrayList<>();
        for (int i = left; i < left + k; i++) {
            res.add(arr[i]);
        }
        return res;
    }

    private int findUpperRight(int[] arr, int left, int right, int x) {
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] >= x) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    private boolean isLeftClose(int[] arr, int left, int right, int x) {
        if (left < 0) {
            return false;
        }
        if (right >= arr.length) {
            return true;
        }
        return x - arr[left] <= arr[right] - x;
    }

    //[662].二叉树最大宽度
    public int widthOfBinaryTree(TreeNode root) {
        if (root == null) return 0;
        AtomicInteger res = new AtomicInteger();
        dfs(root, 0, 0, new HashMap<>(), res);
        return res.get();
    }

    private void dfs(TreeNode root, int depth, int pos, Map<Integer, Integer> map, AtomicInteger res) {
        if (root == null) return;

        map.putIfAbsent(depth, pos);

        res.set(Math.max(res.get(), pos - map.get(depth) + 1));

        dfs(root.left, depth + 1, 2 * pos + 1, map, res);
        dfs(root.right, depth + 1, 2 * pos + 2, map, res);
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

    //[675].为高尔夫比赛砍树
    class Solution675 {
        int N = 50;
        int[][] g = new int[N][N];
        int n, m;
        public int cutOffTree(List<List<Integer>> forest) {
            List<int[]> graph = new ArrayList<>();
            n = forest.size();
            m = forest.get(0).size();
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    g[i][j] = forest.get(i).get(j);
                    if (g[i][j] > 1) {
                        graph.add(new int[] {g[i][j], i, j});
                    }
                }
            }
            if (g[0][0] == 0) return -1;
            Collections.sort(graph, (a,b)->a[0]-b[0]);
            int x = 0, y = 0, ans = 0;
            for (int[] ne : graph) {
                int nx = ne[1], ny = ne[2];
                int d = bfs(x, y, nx, ny);
                if (d == -1) return -1;
                ans += d;
                x = nx; y = ny;
            }
            return ans;
        }

        int [][] directs = new int[][] {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
        int bfs(int x, int y, int nx, int ny) {
            if (x == nx && y == ny) return 0;
            int ans = 0;
            Queue<int[]> queue = new LinkedList<>();
            boolean[][] visited = new boolean[n][m];
            queue.offer(new int[] {x, y});
            visited[x][y] = true;
            while (!queue.isEmpty()) {
                int size = queue.size();
                for (int i = 0; i < size; i++) {
                    int[] cur = queue.poll();
                    int cX = cur[0], cY = cur[1];
                    for (int[] direct :directs) {
                        int nX = cX + direct[0], nY = cY + direct[1];
                        if (nX < 0 || nY <0 || nX >= n || nY >= m) continue;
                        if (g[nX][nY] == 0 || visited[nX][nY]) continue;
                        if (nX == nx && nY == ny) return ans +1;
                        queue.offer(new int[] {nX, nY});
                        visited[nX][nY] = true;
                    }
                }
                ans++;
            }
            return -1;
        }

    }

    //[679].24 点游戏
    public class Solution679 {
        double EPSILON = 1e-6;
        double TARGET = 24;

        public boolean judgePoint24(int[] cards) {

            double[] nums = new double[4];
            for (int i = 0; i < cards.length; i++) {
                nums[i] = cards[i];
            }
            return helper(nums);
        }

        private boolean helper(double[] nums) {
            if (nums.length == 1) return Math.abs(nums[0] - TARGET) < EPSILON;
            for (int i = 0; i < nums.length; i++) {
                for (int j = i + 1; j < nums.length; j++) {
                    double[] temp = new double[nums.length - 1];
                    for (int k = 0, index = 0; k < nums.length; k++)
                        if (k != i && k != j) temp[index++] = nums[k];
                    for (double num : cal(nums[i], nums[j])) {
                        temp[temp.length - 1] = num;
                        if (helper(temp)) {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        private List<Double> cal(double a, double b) {
            List<Double> res = new ArrayList<>();
            res.add(a + b);
            res.add(a - b);
            res.add(b - a);
            res.add(a * b);
            //小于1e-6认为是0
            if (!(Math.abs(a) < EPSILON)) res.add(b / a);
            if (!(Math.abs(b) < EPSILON)) res.add(a / b);
            return res;
        }

    }

    //[682].棒球比赛
    public int calPoints(String[] ops) {
        Stack<Integer> stack = new Stack<>();
        int ans = 0;
        for (String op : ops) {
            if (op.equals("C")) {
                ans -= stack.pop();
            } else if (op.equals("D")) {
                int num = stack.peek() * 2;
                ans += num;
                stack.push(num);
            } else if (op.equals("+")) {
                int first = stack.pop();
                int second = stack.pop();
                int num = first + second;

                ans += num;
                stack.push(second);
                stack.push(first);
                stack.push(num);
            } else {
                int num = Integer.parseInt(op);
                ans += num;
                stack.push(num);
            }
        }
        return ans;
    }

    //[684].冗余连接
    public int[] findRedundantConnection(int[][] edges) {
        int n = edges.length;
        //从1开始
        UnionFind uf = new UnionFind(n + 1);
        for (int[] edge : edges) {
            if (uf.connect(edge[0], edge[1])) {
                return edge;
            }
            uf.union(edge[0], edge[1]);
        }
        return new int[0];
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

    //[688].骑士在棋盘上的概率
    public double knightProbability(int n, int k, int row, int column) {
        //位置i，j上移动k步的概率
        double[][][] dp = new double[n][n][k + 1];
        int[][] directs = new int[][]{{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                dp[i][j][0] = 1;
            }
        }

        for (int p = 1; p <= k; p++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    for (int[] dir : directs) {
                        int nx = i + dir[0];
                        int ny = j + dir[1];
                        if (nx < 0 || ny < 0 || nx >= n || ny >= n) {
                            continue;
                        }
                        dp[i][j][p] += dp[nx][ny][p - 1] / 8.0d;
                    }
                }
            }
        }

        return dp[row][column][k];
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

    //[692].前K个高频单词
    public List<String> topKFrequent(String[] words, int k) {
        Map<String, Integer> countMap = new HashMap<>();
        for (String word : words) countMap.put(word, countMap.getOrDefault(word, 0) + 1);
        //注意这里用的是小根堆，因为是topk，堆顶是最小的，最后的倒序就是从大到小
        PriorityQueue<Object[]> queue = new PriorityQueue<>((a, b) -> a[1] == b[1] ? ((String) b[0]).compareTo((String) a[0]) : (int) a[1] - (int) b[1]);

        for (String word : countMap.keySet()) {
            int count = countMap.get(word);
            queue.offer(new Object[]{word, count});
            if (queue.size() > k) {
                queue.poll();
            }
        }
        List<String> res = new ArrayList<>();
        while (!queue.isEmpty()) {
            res.add((String) queue.poll()[0]);
        }
        Collections.reverse(res);
        return res;
    }

    //[693].交替位二进制数
    public boolean hasAlternatingBits(int n) {
        //交替二进制，右移一位，异或必定为0000011111这种
        int x = n & n >> 1;
        return (x & (x + 1)) == 0;
    }

    //[694].不同岛屿的数量
    public int numDistinctIslands(int[][] grid) {
        Set<String> islands = new HashSet<>();
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    StringBuilder sb = new StringBuilder();
                    dfsForNumDistinctIslands(grid, i, j, i, j, sb);
                    islands.add(sb.toString());
                }
            }
        }
        return islands.size();
    }

    private void dfsForNumDistinctIslands(int[][] grid, int x, int y, int originalX, int originalY, StringBuilder sb) {
        if (x < 0 || y < 0 || x >= grid.length || y >= grid[0].length || grid[x][y] == 0) {
            return;
        }
        grid[x][y] = 0;
        //因为多源dfs，肯定是从左上角开始的，所以可以记录相对位置坐标
        sb.append(originalX - x);
        sb.append(originalY - y);
        dfsForNumDistinctIslands(grid, x + 1, y, originalX, originalY, sb);
        dfsForNumDistinctIslands(grid, x - 1, y, originalX, originalY, sb);
        dfsForNumDistinctIslands(grid, x, y + 1, originalX, originalY, sb);
        dfsForNumDistinctIslands(grid, x, y - 1, originalX, originalY, sb);
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

    //[698].划分为k个相等的子集
    public static boolean canPartitionKSubsets(int[] nums, int k) {
        if (k > nums.length) return false;
        int sum = Arrays.stream(nums).sum();
        if (sum % k != 0) return false;
        sum = sum / k;
        int used = 0;
        Map<Integer, Boolean> memo = new HashMap<>();
        return dfsForCanPartitionKSubsets(nums, 0, 0, sum, used, k, memo);
    }

    private static boolean dfsForCanPartitionKSubsets(int[] nums, int start, int bucket, int target, int used, int k, Map<Integer, Boolean> memo) {
        if (k == 0) {
            return true;
        }

        if (bucket == target) {
            boolean result = dfsForCanPartitionKSubsets(nums, 0, 0, target, used, k - 1, memo);
            memo.put(used, result);
            return result;
        }

        if (memo.containsKey(used)) {
            return memo.get(used);
        }

        for (int i = start; i < nums.length; i++) {
            //使用过了，剪枝
            if (((used >> i) & 1) == 1) {
                continue;
            }
            //放不下，剪枝
            if (bucket + nums[i] > target) {
                continue;
            }
            used |= 1 << i;
            bucket += nums[i];

            boolean result = dfsForCanPartitionKSubsets(nums, i + 1, bucket, target, used, k, memo);
            if (result) {
                return true;
            }

            used ^= 1 << i;
            bucket -= nums[i];
        }
        return false;
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
//        TreeNode parent = null, cur = root;
//        while (cur != null) {
//            parent = cur;
//            if (cur.val < val) {
//                cur = cur.right;
//            } else {
//                cur = cur.left;
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

    //[706].设计哈希映射
    public class MyHashMap {

        class Node {
            private int key;
            private int value;
            private Node next;

            public Node(int key, int value) {
                this.key = key;
                this.value = value;
            }
        }

        private int BASE = 1009;
        private Node[] nodes;

        public MyHashMap() {
            nodes = new Node[BASE];
        }

        public void put(int key, int value) {
            int idx = getIndex(key);
            Node loc = nodes[idx], tmp = loc;
            if (loc != null) {
                while (tmp != null) {
                    if (tmp.key == key) {
                        tmp.value = value;
                        return;
                    }
                    tmp = tmp.next;
                }
            }
            //头插法
            Node newOne = new Node(key, value);
            newOne.next = loc;
            nodes[idx] = newOne;
        }

        public int get(int key) {
            int idx = getIndex(key);
            Node loc = nodes[idx];
            if (loc != null) {
                while (loc != null) {
                    if (loc.key == key) {
                        return loc.value;
                    }
                    loc = loc.next;
                }
            }
            return -1;
        }

        public void remove(int key) {
            int idx = getIndex(key);
            Node loc = nodes[idx];
            if (loc != null) {
                Node pre = null;
                while (loc != null) {
                    if (loc.key == key) {
                        if (pre != null) {
                            pre.next = loc.next;
                        } else {
                            nodes[idx] = loc.next;
                        }
                        return;
                    }
                    pre = loc;
                    loc = loc.next;
                }
            }
        }

        private int getIndex(int key) {
            int hash = Integer.hashCode(key);
            hash ^= (hash >>> 16);
            return hash % BASE;
        }
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

    //[713].乘积小于K的子数组
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if (k <= 1) return 0;
        int n = nums.length;
        int prod = 1, ans = 0;
        for (int l = 0, r = 0; r < n; r++) {
            prod *= nums[r];
            while (prod >= k) {
                prod /= nums[l++];
            }
            ans += r - l + 1;
        }
        return ans;
    }

    //[714].买卖股票的最佳时机含手续费
    public int maxProfit(int[] prices, int fee) {
        int n = prices.length;
        int[][] dp = new int[n][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for (int i = 1; i < n; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }
        return dp[n - 1][0];
    }

    //[717].1比特与2比特字符
    public boolean isOneBitCharacter(int[] bits) {
        int idx = 0, n = bits.length;
        while (idx < n - 1) {
            if (bits[idx] == 0) {
                idx++;
            } else {
                idx += 2;
            }
        }
        return idx == n - 1;
    }

    //[718].最长重复子数组
    public int findLength(int[] nums1, int[] nums2) {
//        int m = nums1.length, n = nums2.length;
//        //长度为i的nums1，与长度为j的nums2构成的相同子数组的最大长度
//        int[][] dp = new int[m + 1][n + 1];
//        int ans = 0;
//        for (int i = 1; i <= m; i++) {
//            for (int j = 1; j <= n; j++) {
//                if (nums1[i - 1] == nums2[j - 1]) {
//                    dp[i][j] = dp[i - 1][j - 1] + 1;
//                }
//
//                ans = Math.max(ans, dp[i][j]);
//            }
//        }
//        return ans;
        int m = nums1.length, n = nums2.length;
        int[] dp = new int[n + 1];
        int ans = 0;
        for (int i = 1; i <= m; i++) {
            for (int j = n; j >= 1; j--) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[j] = dp[j - 1] + 1;
                } else {
                    //二维转一维需要注意这个点！！！！！
                    dp[j] = 0;
                }
                ans = Math.max(ans, dp[j]);
            }
        }
        return ans;
    }

    //[720].词典中最长的单词
    public String longestWord(String[] words) {
        Set<String> set = new HashSet<>();
        for (String word : words) set.add(word);
        String ans = "";
        out:
        for (String word : set) {
            int m = word.length(), n = ans.length();
            if (m < n) continue;
            if (m == n && word.compareTo(ans) >= 0) continue;

            for (int i = 1; i <= word.length(); i++) {
                if (!set.contains(word.substring(0, i))) {
                    continue out;
                }
            }
            ans = word;
        }
        return ans;
    }

    //[735].行星碰撞
    public static int[] asteroidCollision(int[] asteroids) {
        int n = asteroids.length;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < n; i++) {
            int num = asteroids[i];
            boolean push = true;
            while (!stack.isEmpty()) {
                int top = asteroids[stack.peek()];
                if (top > 0 && num < 0) {
                    if (top < -num) {
                        stack.pop();
                    } else if (top == -num) {
                        stack.pop();
                        push = false;
                        break;
                    } else {
                        push = false;
                        break;
                    }
                } else {
                    break;
                }
            }
            if (push) {
                stack.push(i);
            }
        }
        int[] res = new int[stack.size()];
        int index = res.length - 1;
        while (!stack.isEmpty()) {
            res[index--] = asteroids[stack.pop()];
        }
        return res;
    }

    //[739].每日温度
    public int[] dailyTemperatures(int[] temperatures) {
        int[] res = new int[temperatures.length];
        Stack<Integer> stack = new Stack<>();
        //单调递增减栈，当前元素大于栈顶元素则压掉它
        for (int i = temperatures.length - 1; i >= 0; i--) {
            //47 47 46 76 如果没有=，对第一个47而言认为比第二个47比它大
            while (!stack.isEmpty() && temperatures[i] >= temperatures[stack.peek()]) {
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
            for (int[] next : graph[id]) {
                int nextId = next[0];
                int weight = next[1];
                int distToNext = distTo[id] + weight;
                if (distToNext < distTo[nextId]) {
                    distTo[nextId] = distToNext;
                    queue.offer(new int[]{nextId, distToNext});
                }
            }
        }
        return distTo;
    }

    //[744].寻找比目标字母大的最小字母
    public char nextGreatestLetter(char[] letters, char target) {
        int n = letters.length;
        int left = 0, right = n - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (letters[mid] <= target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        if (letters[left] <= target) {
            return letters[0];
        } else {
            return letters[left];
        }
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

    //[752].打开转盘锁
    public int openLock(String[] deadends, String target) {
        String start = "0000";
        if (start.equals(target)) return 0;
        Set<String> deads = new HashSet<>();
        for (String d : deadends) deads.add(d);

        if (deads.contains(start)) return -1;
        Set<String> q1 = new HashSet<>();
        Set<String> q2 = new HashSet<>();
        Set<String> visited = new HashSet<>();
        q1.add(start);
        q2.add(target);
        int step = 0;
        while (!q1.isEmpty() && !q2.isEmpty()) {
            //增加新的，防止q1被污染
            Set<String> temp = new HashSet<>();
            for (String cur : q1) {
                if (deads.contains(cur)) {
                    continue;
                }
                //只要两个队列中包含了，就要返回
                if (q2.contains(cur)) {
                    return step;
                }

                visited.add(cur);
                for (int i = 0; i < 4; i++) {
                    String up = plusOne(cur, i);
                    if (!visited.contains(up)) {
                        temp.add(up);
                    }
                    String down = minusOne(cur, i);
                    if (!visited.contains(down)) {
                        temp.add(down);
                    }
                }
            }
            //没扩展一层，增加一个
            step++;
            //交换队列
            q1 = q2;
            //增加新的
            q2 = temp;
        }
        return -1;
    }

    private String plusOne(String cur, int i) {
        char[] arr = cur.toCharArray();
        arr[i] = (arr[i] == '9') ? '0' : (char) (arr[i] + 1);
        return String.valueOf(arr);
    }

    private String minusOne(String cur, int i) {
        char[] arr = cur.toCharArray();
        arr[i] = (arr[i] == '0') ? '9' : (char) (arr[i] - 1);
        return String.valueOf(arr);
    }

    public class Solution759 {
        class Interval {
            int start;
            int end;

            public Interval(int start, int end) {
                this.start = start;
                this.end = end;
            }
        }

        public List<Interval> employeeFreeTime(List<List<Interval>> schedule) {
            List<Interval> all = new ArrayList<>();
            for (List<Interval> intervals : schedule) {
                for (Interval interval : intervals) {
                    all.add(interval);
                }
            }
            Collections.sort(all, (a, b) -> a.start - b.start);
            Interval first = all.get(0);
            int end = first.end;
            List<Interval> res = new ArrayList<>();
            for (int i = 1; i < all.size(); i++) {
                Interval cur = all.get(i);
                //没有合并
                if (cur.start > end) {
                    res.add(new Interval(end, cur.start));
                }
                end = Math.max(cur.end, end);
            }
            return all;
        }
    }


    //[767].重构字符串
    public String reorganizeString(String s) {
        int[] cnt = new int[26];
        for (char ch : s.toCharArray()) {
            cnt[ch - 'a']++;
        }
        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> b[1] - a[1]);
        for (int i = 0; i < 26; i++) {
            if (cnt[i] != 0) {
                queue.offer(new int[]{i, cnt[i]});
            }
        }

        StringBuilder sb = new StringBuilder();
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            char ch = (char) (cur[0] + 'a');
            if (sb.length() > 0 && sb.charAt(sb.length() - 1) == ch) {
                if (!queue.isEmpty() && queue.peek()[1] > 0) {
                    sb.append((char) (queue.peek()[0] + 'a'));
                    if (--queue.peek()[1] == 0) {
                        queue.poll();
                    }
                    queue.offer(cur);
                } else {
                    return "";
                }
            } else {
                sb.append(ch);
                if (--cur[1] > 0) {
                    queue.offer(cur);
                }
            }
        }
        return sb.toString();
    }

    //[780].到达终点
    public boolean reachingPoints(int sx, int sy, int tx, int ty) {
        // 1,1 -> 1,2 -> 3,2 -> 3,5
        //正推方向有两个，反推方向就一个， 如果tx==ty，那么前一个状态是0,0，非法状态。
        while (tx >= sx && ty >= sy) {
            if (tx == sx && ty == sy) return true;
            //用来加速判断（1,1 -> 1,2）
            if (tx == sx && ty > sy && (ty - sy) % sx == 0) return true;
            //用来加速判断（1,2 -> 3,2）
            if (ty == sy && tx > sx && (tx - sx) % sy == 0) return true;
            //本来是辗转相减，但是为了缩减次数换用取余
            if (tx > ty) {
                tx %= ty;
            } else {
                ty %= tx;
            }
        }
        return false;
    }

    //[785].判断二分图
    public class Solution785 {
        int[] color;
        int RED = 1;
        int GREEN = 2;
        int UNCOLORED = 0;
        boolean valid;

        public boolean isBipartite(int[][] graph) {
            int n = graph.length;
            color = new int[n];
            valid = true;

            for (int i = 0; i < n && valid; i++) {
                if (color[i] == UNCOLORED) {
                    dfs(graph, i, RED);
                }
            }
            return valid;
        }

        private void dfs(int[][] graph, int i, int c) {
            color[i] = c;
            int neColor = c == RED ? GREEN : RED;
            for (int ne : graph[i]) {
                if (color[ne] == UNCOLORED) {
                    dfs(graph, ne, neColor);
                    if (!valid) {
                        return;
                    }
                } else {
                    if (color[ne] != neColor) {
                        valid = false;
                        return;
                    }
                }
            }
        }
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

    //[798].得分最高的最小轮调
    public class Solution798 {

        int N = 100010;
        int[] diff = new int[N];

        private void add(int l, int r) {
            diff[l] += 1;
            diff[r + 1] -= 1;
        }

        public int bestRotation(int[] nums) {
            Arrays.fill(diff, 0);
            int n = nums.length;
            //这个题得逆向思维，对于每个元素而言，合法的k的取之范围是什么，统计不同k值合法的元素个数，最大值就是求解的k
            //i-k为新下标，取值范围为[0, n-1]
            //i -k >=0, i - k <= n-1
            //nums[i]<= i - k
            //因此nums[i] 能够得分的 k 的取值范围为 [i - (n - 1), i - nums[i]]。
            //另 a = i - (n - 1); b = i - nums[i]; 如果a <=b, 则k的区间都得+1， 如果a > b，则k的区间[0,b]和[a,n-1]都+1
            for (int i = 0; i < n; i++) {
                int a = (i - (n - 1) + n) % n;
                int b = (i - nums[i] + n) % n;
                if (a <= b) {
                    add(a, b);
                } else {
                    add(0, b);
                    add(a, n - 1);
                }
            }

            for (int i = 1; i <= n; i++) diff[i] += diff[i - 1];
            int ans = 0;
            for (int k = 1; k <= n; k++) {
                if (diff[k] > diff[ans]) {
                    ans = k;
                }
            }
            return ans;
        }
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

    //[804].唯一摩尔斯密码词
    public int uniqueMorseRepresentations(String[] words) {
        String[] codes = {".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--", "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."};
        Set<String> set = new HashSet<>();
        for (String word : words) {
            StringBuilder sb = new StringBuilder();
            for (char ch : word.toCharArray()) {
                sb.append(codes[ch - 'a']);
            }
            set.add(sb.toString());
        }
        return set.size();
    }

    //[806].写字符串需要的行数
    public int[] numberOfLines(int[] widths, String s) {
        int line = 1, count = 0;
        for (char ch : s.toCharArray()) {
            int cnt = widths[ch - 'a'];
            if (count + cnt > 100) {
                count = cnt;
                line++;
            } else {
                count += cnt;
            }
        }
        return new int[]{line, count};
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

    //[819].最常见的单词
    public String mostCommonWord(String paragraph, String[] banned) {
        Map<String, Integer> cnt = new HashMap<>();
        String mostCommon = "";
        int maxCnt = 0;
        int n = paragraph.length();
        char[] cs = paragraph.toCharArray();
        Set<String> bannedSet = new HashSet<>();
        for (String str : banned) {
            bannedSet.add(str.toLowerCase());
        }
        for (int i = 0; i < n; ) {
            if (!Character.isLetter(cs[i]) && ++i > 0) continue;

            int j = i;
            while (j < n && Character.isLetter(cs[j])) {
                j++;
            }

            String key = paragraph.substring(i, j).toLowerCase();
            if (!bannedSet.contains(key)) {
                cnt.put(key, cnt.getOrDefault(key, 0) + 1);
                if (cnt.get(key) > maxCnt) {
                    mostCommon = key;
                    maxCnt = cnt.get(key);
                }
            }
            i = j + 1;
        }
        return mostCommon;
    }

    //[821].字符的最短距离
    public int[] shortestToChar(String s, char c) {
        int n = s.length();
        int[] res = new int[n];
        Arrays.fill(res, n + 1);
        char[] cs = s.toCharArray();
        for (int i = 0, j = -1; i < n; i++) {
            if (cs[i] == c) {
                j = i;
            }
            if (j != -1) {
                res[i] = i - j;
            }
        }
        for (int i = n - 1, j = -1; i >= 0; i--) {
            if (cs[i] == c) {
                j = i;
            }
            if (j != -1) {
                res[i] = Math.min(res[i], j - i);
            }
        }
        return res;
    }

    //[824].山羊拉丁文
    public String toGoatLatin(String sentence) {
        int n = sentence.length();
        char[] cs = sentence.toCharArray();
        StringBuilder sb = new StringBuilder();
        String last = "a";
        for (int i = 0; i < n; ) {
            int j = i;
            while (j < n && cs[j] != ' ') j++;
            if ("aeiouAEIOU".indexOf(cs[i]) == -1) {
                sb.append(sentence.substring(i + 1, j)).append(cs[i]);
            } else {
                sb.append(sentence.substring(i, j));
            }
            sb.append("ma").append(last);

            last += "a";
            i = j + 1;
            if (i < n) {
                sb.append(' ');
            }
        }
        return sb.toString();
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

    //[838].推多米诺
    public String pushDominoes(String dominoes) {
        int n = dominoes.length();
        char[] s = dominoes.toCharArray();
        char left = 'L';
        int i = 0;
        while (i < n) {
            int j = i;
            while (j < n && s[j] == '.') j++;
            //左边是L，右边是R，不受影响
            char right = j < n ? s[j] : 'R';
            //...L往左边倒， ...R往右边倒，直到右边有R的时候再处理
            if (left == right) {
                while (i < j) {
                    s[i++] = left;
                }
            } else if (left == 'R' && right == 'L') {
                //相向倒
                int k = j - 1;
                while (i < k) {
                    s[i++] = 'R';
                    s[k--] = 'L';
                }
            }

            i = j + 1;
            left = right;
        }
        return new String(s);
    }

    //[844].比较含退格的字符串
    public static boolean backspaceCompare(String s, String t) {
        int i = s.length() - 1, j = t.length() - 1;
        int skipS = 0, skipT = 0;
        while (i >= 0 || j >= 0) {
            while (i >= 0) {
                char ch = s.charAt(i);
                if (ch == '#') {
                    skipS++;
                    i--;
                } else if (skipS > 0) {
                    skipS--;
                    i--;
                } else {
                    //需要对比的数据
                    break;
                }
            }
            while (j >= 0) {
                char ch = t.charAt(j);
                if (ch == '#') {
                    skipT++;
                    j--;
                } else if (skipT > 0) {
                    skipT--;
                    j--;
                } else {
                    //需要对比的数据
                    break;
                }
            }
            if (i >= 0 && j >= 0) {
                if (s.charAt(i) != t.charAt(j)) return false;
            } else if (i >= 0 || j >= 0) {
                return false;
            }
            i--;
            j--;
        }
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

    //[848].字母移位
    public String shiftingLetters(String s, int[] shifts) {
        int n = s.length();
        long sum = 0L;
        //比较明显是个前缀和做法，i的字母会受到shifts i...n-1的影响，
        //那么从后遍历，计算出当前字母需要变化的次数，根据26取模运算当前字符。
        char[] arr = s.toCharArray();
        for (int i = n - 1; i >= 0; i--) {
            sum += shifts[i];
            arr[i] = (char) ((arr[i] + sum - 'a') % 26 + 'a');
        }
        return String.valueOf(arr);
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

    //[863].二叉树中所有距离为 K 的结点
    public List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
        //距离的问题，一定要优先想到BFS
        Map<Integer, TreeNode> parents = new HashMap<>();
        dfsForDistanceK(root, null, parents);

        Queue<TreeNode> queue = new ArrayDeque<>();
        Set<TreeNode> visited = new HashSet<>();
        queue.offer(target);
        visited.add(target);
        int dis = 0;
        List<Integer> res = new ArrayList<>();
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();
                if (dis == k) {
                    res.add(cur.val);
                } else if (dis < k) {
                    if (cur.left != null && !visited.contains(cur.left)) {
                        visited.add(cur.left);
                        queue.offer(cur.left);
                    }
                    if (cur.right != null && !visited.contains(cur.right)) {
                        visited.add(cur.right);
                        queue.offer(cur.right);
                    }
                    TreeNode parent = parents.get(cur.val);
                    if (parent != null && !visited.contains(parent)) {
                        visited.add(parent);
                        queue.offer(parent);
                    }
                }
            }
            dis++;
        }
        return res;
    }

    private void dfsForDistanceK(TreeNode root, TreeNode parent, Map<Integer, TreeNode> parents) {
        if (root == null) return;
        parents.put(root.val, parent);
        dfsForDistanceK(root.left, root, parents);
        dfsForDistanceK(root.right, root, parents);
    }

    //[867].转置矩阵
    public int[][] transpose(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[][] res = new int[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                res[j][i] = matrix[i][j];
            }
        }
        return res;
    }

    //[868].二进制间距
    public int binaryGap(int n) {
        int index = -1, maxAns = 0;
        for (int i = 0; i < 32; i++) {
            if ((n >> i & 1) == 1) {
                if (index != -1) {
                    maxAns = Math.max(i - index, maxAns);
                }
                index = i;
            }
        }
        return maxAns;
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

    //[877].石子游戏
    public boolean stoneGame(int[] piles) {
        //此题跟上面的486.预测赢家一样的解法
        int n = piles.length;
        //dp[i][j] i...j 先手领先后手的最大分差
        int[][] dp = new int[n][n];
        //只有一个元素的时候，最大分差就是本身。
        for (int i = 0; i < n; i++) {
            dp[i][i] = piles[i];
        }

        //依赖下和左，所以倒序遍历，从下到上，从左往右
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                dp[i][j] = Math.max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1]);
            }
        }
        //总分值为奇数，所以不可能出现=0的情况
        return dp[0][n - 1] > 0;
    }

    //[883].三维形体投影面积
    public int projectionArea(int[][] grid) {
        int n = grid.length;
        int ans = 0;
        for (int i = 0; i < n; i++) {
            int maxCol = 0, maxRow = 0;
            for (int j = 0; j < n; j++) {
                maxCol = Math.max(maxCol, grid[i][j]);
                maxRow = Math.max(maxRow, grid[j][i]);

                if (grid[i][j] > 0) {
                    ans++;
                }
            }
            ans += maxCol;
            ans += maxRow;
        }
        return ans;
    }

    //[884].两句话中的不常见单词
    public String[] uncommonFromSentences(String s1, String s2) {
        Map<String, Integer> count = new HashMap<>();
        String[] sp1 = s1.split(" ");
        String[] sp2 = s2.split(" ");
        for (String sp : sp1) {
            count.put(sp, count.getOrDefault(sp, 0) + 1);
        }
        for (String sp : sp2) {
            count.put(sp, count.getOrDefault(sp, 0) + 1);
        }
        List<String> temp = new ArrayList<>();
        for (String key : count.keySet()) {
            if (count.get(key) == 1) {
                temp.add(key);
            }
        }
        return temp.toArray(new String[0]);
    }

    //[895].最大频率栈
    public class FreqStack {
        Map<Integer, Integer> numberFreq;
        Map<Integer, Stack<Integer>> freqNumber;
        int maxFreq;

        public FreqStack() {
            numberFreq = new HashMap<>();
            freqNumber = new HashMap<>();
            maxFreq = 0;
        }

        public void push(int val) {
            int freq = numberFreq.getOrDefault(val, 0) + 1;
            numberFreq.put(val, freq);

            freqNumber.putIfAbsent(freq, new Stack<>());
            Stack<Integer> stack = freqNumber.get(freq);
            stack.push(val);

            maxFreq = Math.max(maxFreq, freq);
        }

        public int pop() {
            Stack<Integer> stack = freqNumber.get(maxFreq);
            int number = stack.pop();
            numberFreq.put(number, numberFreq.get(number) - 1);

            if (stack.isEmpty()) {
                maxFreq--;
            }
            return number;
        }
    }

    //[907].子数组的最小值之和
    public int sumSubarrayMins(int[] arr) {
        if (arr.length == 1) return arr[0];
        int BASE = (int) 1e9 + 7;
        Stack<Integer> stack = new Stack<>();
        long res = 0;
        for (int i = 0; i < arr.length; i++) {
            while (!stack.isEmpty() && arr[i] <= arr[stack.peek()]) {
                int index = stack.pop();
                int preIndex = stack.isEmpty() ? -1 : stack.peek();
                int preCount = index - preIndex;
                int nextCount = i - index;

                res += (long) arr[index] * preCount * nextCount;
                res %= BASE;
            }
            stack.push(i);
        }
        while (!stack.isEmpty()) {
            int index = stack.pop();
            int preIndex = stack.isEmpty() ? -1 : stack.peek();

            res += (long) arr[index] * (index - preIndex) * (arr.length - index);
            res %= BASE;
        }
        return (int) (res % BASE);
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

    //[917].仅仅反转字母
    public String reverseOnlyLetters(String s) {
        int left = 0, right = s.length() - 1;
        char[] arr = s.toCharArray();
        while (left < right) {
            if (!Character.isLetter(arr[left])) {
                left++;
            } else if (!Character.isLetter(arr[right])) {
                right--;
            } else {
                char temp = arr[left];
                arr[left] = arr[right];
                arr[right] = temp;
                left++;
                right--;
            }
        }
        return String.valueOf(arr);
    }

    //[937].重新排列日志文件
    public String[] reorderLogFiles(String[] logs) {
        int n = logs.length;
        Log[] arr = new Log[n];
        for (int i = 0; i < n; i++) {
            arr[i] = new Log(logs[i], i);
        }

        Arrays.sort(arr, (a, b) -> {
            if (a.type != b.type) return a.type - b.type;
            else if (a.type == 1) return a.index - b.index;
            else return a.content.compareTo(b.content) == 0 ? a.sign.compareTo(b.sign) : a.content.compareTo(b.content);
        });

        String[] res = new String[n];
        for (int i = 0; i < n; i++) {
            res[i] = arr[i].orig;
        }
        return res;
    }

    class Log {
        String orig, content, sign;
        int index;
        int type;

        public Log(String log, int index) {
            this.index = index;
            orig = log;
            int n = log.length(), i = 0;
            while (i < n && log.charAt(i) != ' ') i++;
            sign = log.substring(0, i);
            content = log.substring(i + 1);
            type = Character.isDigit(content.charAt(0)) ? 1 : 0;
        }
    }

    //[949].给定数字能组成的最大时间
    public String largestTimeFromDigits(int[] arr) {
        String ans = "";
        Arrays.sort(arr);
        for (int i = 3; i >= 0; i--) {
            if (arr[i] > 2) continue;
            for (int j = 3; j >= 0; j--) {
                if (i == j || arr[i] == 2 && arr[j] > 3) continue;
                for (int k = 3; k >= 0; k--) {
                    if (k == i || k == j || arr[k] > 5) continue;
                    return "" + arr[i] + arr[j] + ":" + arr[k] + arr[6 - i - j - k];
                }
            }
        }
        return "";
    }

    //[952].按公因数计算最大组件大小
    public int largestComponentSize(int[] nums) {
        int max = Arrays.stream(nums).max().getAsInt();
        UnionFind4 uf = new UnionFind4(max + 1);
        for (int num : nums) {
            double up = Math.sqrt(num);
            for (int i = 2; i <= up; i++) {
                if (num % i == 0) {
                    uf.union(num, i);
                    uf.union(num, num / i);
                }
            }
        }

        //因为通过了质因子作为中间桥梁，实际只会算数字的连通分量，故常规并查集直接求连通分量是不行的
        //通过遍历每个数的根节点来计数可以排除掉质因子的数量
        int ans = 0;
        int[] cnt = new int[max + 1];
        for (int num : nums) {
            int root = uf.find(num);
            cnt[root]++;
            ans = Math.max(ans, cnt[root]);
        }
        return ans;
    }

    public static class UnionFind4 {
        private int[] parent;

        public UnionFind4(int size) {
            parent = new int[size];
            for (int i = 0; i < size; i++) {
                parent[i] = i;
            }
        }

        public void union(int p, int q) {
            int rootP = find(p);
            int rootQ = find(q);
            if (rootQ == rootP) return;
            parent[rootP] = rootQ;
        }

        public int find(int x) {
            while (x != parent[x]) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }

        public boolean connected(int p, int q) {
            int rootP = find(p);
            int rootQ = find(q);
            return rootP == rootQ;
        }
    }

    //[953].验证外星语词典
    public boolean isAlienSorted(String[] words, String order) {
        int[] index = new int[26];
        for (int i = 0; i < order.length(); i++) {
            index[order.charAt(i) - 'a'] = i;
        }

        for (int i = 0; i < words.length - 1; i++) {
            if (!valid(words[i], words[i + 1], index)) {
                return false;
            }
        }
        return true;
    }

    private boolean valid(String first, String second, int[] index) {
        int m = first.length(), n = second.length();
        int i = 0;
        while (i < m && i < n && first.charAt(i) == second.charAt(i)) i++;

        if (i < m && i < n) {
            return index[first.charAt(i) - 'a'] < index[second.charAt(i) - 'a'];
        } else if (i == m) {
            return true;
        } else {
            return false;
        }
    }

    //[954].二倍数对数组
    public boolean canReorderDoubled(int[] arr) {
        Map<Integer, Integer> cnt = new HashMap<Integer, Integer>();
        for (int x : arr) {
            cnt.put(x, cnt.getOrDefault(x, 0) + 1);
        }
        if (cnt.getOrDefault(0, 0) % 2 != 0) {
            return false;
        }

        List<Integer> vals = new ArrayList<Integer>();
        for (int x : cnt.keySet()) {
            vals.add(x);
        }
        Collections.sort(vals, (a, b) -> Math.abs(a) - Math.abs(b));

        for (int x : vals) {
            if (cnt.getOrDefault(2 * x, 0) < cnt.get(x)) { // 无法找到足够的 2x 与 x 配对
                return false;
            }
            cnt.put(2 * x, cnt.getOrDefault(2 * x, 0) - cnt.get(x));
        }
        return true;
    }

    //[968].监控二叉树
    public class Solution968 {
        int uncovered = 0;
        int camera = 1;
        int covered = 2;

        int result = 0;

        public int minCameraCover(TreeNode root) {
            //根节点未覆盖，那么就需要在设置一个
            if (dfs(root) == uncovered) {
                result++;
            }
            return result;
        }

        private int dfs(TreeNode root) {
            //空节点，得保证叶子节点没有覆盖，从而在叶子父节点上设置摄像头，所以空节点为覆盖。
            //当然也不能设置为摄像头，贪心嘛
            if (root == null) {
                return covered;
            }

            int left = dfs(root.left);
            int right = dfs(root.right);
            // 0    0   1
            // 0    1   1
            // 0    2   1
            // 1    1   2
            // 1    2   2
            // 2    0   1
            // 2    1   2
            // 2    2   0
            //总共9种状态， 都覆盖的时候，考虑父节点设摄像头；无覆盖的时候，需要摄像头；剩下的至少有一个摄像头的时候，为覆盖状态
            //左右都覆盖，那么最好是在父亲节点上设置摄像头，当前节点设置为无覆盖
            if (left == covered && right == covered) return uncovered;

            //至少一个无覆盖
            if (left == uncovered || right == uncovered) {
                result++;
                return camera;
            }

            //至少一个有摄像头
            if (left == camera || right == camera) {
                return covered;
            }

            return -1;
        }
    }

    //[969].煎饼排序
    public List<Integer> pancakeSort(int[] arr) {
        int n = arr.length;
        List<Integer> res = new ArrayList<>();
        for (int i = n; i >= 1; i--) {
            if (i == arr[i - 1]) continue;
            int j = i - 1;
            for (; j >= 1; j--) {
                if (i == arr[j - 1]) {
                    break;
                }
            }

            res.add(j);
            reversePancake(arr, j - 1);
            res.add(i);
            reversePancake(arr, i - 1);
        }
        return res;
    }

    private void reversePancake(int[] arr, int end) {
        int start = 0;
        while (start < end) {
            int temp = arr[start];
            arr[start] = arr[end];
            arr[end] = temp;
            start++;
            end--;
        }
    }

    //[974].和可被 K 整除的子数组
    public static int subarraysDivByK(int[] nums, int k) {
        //题目跟求和为k的子数组个数一样
        //(pre[j] - pre[i]) % k == 0 为区间和是否能被k整除
        //同余定理 => 判断前缀和求余相等即可

        Map<Integer, Integer> preSumCount = new HashMap<>();
        preSumCount.put(0, 1);
        int sum = 0, ans = 0;
        for (int num : nums) {
            sum += num;
            int mod = (sum % k + k) % k;
            if (preSumCount.containsKey(mod)) {
                ans += preSumCount.get(mod);
            }
            preSumCount.put(mod, preSumCount.getOrDefault(mod, 0) + 1);
        }
        return ans;
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

    //[990].等式方程的可满足性
    public boolean equationsPossible(String[] equations) {
        UnionFind uf = new UnionFind(26);
        for (String equation : equations) {
            if (equation.charAt(1) == '=') {
                uf.union(equation.charAt(0) - 'a', equation.charAt(3) - 'a');
            }
        }

        for (String equation : equations) {
            if (equation.charAt(1) == '!') {
                if (uf.connect(equation.charAt(0) - 'a', equation.charAt(3) - 'a')) {
                    return false;
                }
            }
        }
        return true;
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

    //[1001].网格照明
    public int[] gridIllumination(int n, int[][] lamps, int[][] queries) {
        long N = n;
        Set<Long> set = new HashSet<>();
        Map<Integer, Integer> row = new HashMap<>(), col = new HashMap<>();
        Map<Integer, Integer> left = new HashMap<>(), right = new HashMap<>();
        for (int[] lamp : lamps) {
            int x = lamp[0], y = lamp[1];
            int a = x + y, b = x - y;
            long hash = x * N + y;
            if (set.contains(hash)) continue;
            increment(row, x);
            increment(col, y);
            increment(left, a);
            increment(right, b);
            set.add(hash);
        }

        int[][] direct = new int[][]{{0, 0}, {0, 1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {1, -1}, {1, 0}, {1, 1}};
        int[] ans = new int[queries.length];
        for (int i = 0; i < queries.length; i++) {
            int[] query = queries[i];
            int x = query[0], y = query[1];
            int a = x + y, b = x - y;
            if (row.containsKey(x) || col.containsKey(y) || left.containsKey(a) || right.containsKey(b)) {
                ans[i] = 1;
            }
            for (int[] dir : direct) {
                int nx = x + dir[0], ny = y + dir[1];
                if (nx < 0 || ny < 0 || nx >= n || ny >= n) continue;
                int na = nx + ny, nb = nx - ny;
                long hash2 = nx * N + ny;
                if (set.contains(hash2)) {
                    set.remove(hash2);
                    decrement(row, nx);
                    decrement(col, ny);
                    decrement(left, na);
                    decrement(right, nb);
                }
            }
        }
        return ans;
    }

    private void increment(Map<Integer, Integer> map, int key) {
        map.put(key, map.getOrDefault(key, 0) + 1);
    }

    private void decrement(Map<Integer, Integer> map, int key) {
        if (map.get(key) == 1) map.remove(key);
        else map.put(key, map.get(key) - 1);
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

    //[1019].链表中的下一个更大节点
    public static int[] nextLargerNodes(ListNode head) {
        List<Integer> list = new ArrayList<>();
        while (head != null) {
            list.add(head.val);
            head = head.next;
        }

        int[] res = new int[list.size()];
        Stack<Integer> stack = new Stack<>();
        for (int i = list.size() - 1; i >= 0; i--) {
            while (!stack.isEmpty() && list.get(stack.peek()) < list.get(i)) {
                stack.pop();
            }
            res[i] = stack.isEmpty() ? 0 : list.get(stack.peek());
            stack.push(i);
        }
        return res;
    }

    //[1020].飞地的数量
    public int numEnclaves(int[][] grid) {
        int[][] directs = new int[][]{{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
        int m = grid.length, n = grid[0].length;
        for (int i = 0; i < m; i++) {
            dfsForNumEnclaves(grid, i, 0, directs);
            dfsForNumEnclaves(grid, i, n - 1, directs);
        }

        for (int i = 1; i < n - 1; i++) {
            dfsForNumEnclaves(grid, 0, i, directs);
            dfsForNumEnclaves(grid, m - 1, i, directs);
        }
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    res++;
                }
            }
        }
        return res;
    }

    private void dfsForNumEnclaves(int[][] grid, int x, int y, int[][] directs) {
        int m = grid.length, n = grid[0].length;
        if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == 0) return;
        grid[x][y] = 0;

        for (int[] direct : directs) {
            int nx = x + direct[0];
            int ny = y + direct[1];
            dfsForNumEnclaves(grid, nx, ny, directs);
        }
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

    private boolean check(int[] source, int[] target, int n, Set<Long> blocked) {
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
                if (blocked.contains(hash)) continue;
                if (set.contains(hash)) continue;
                set.add(hash);
                queue.offer(new int[]{nextX, nextY});
            }
        }
        return set.size() > n;
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

    //[1060].有序数组中的缺失元素
    public static int missingElement(int[] nums, int k) {
        int n = nums.length;
        int left = 0, right = n - 1;
        if (k > missingCount(nums, n - 1)) {
            return nums[n - 1] + k - missingCount(nums, n - 1);
        }
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (missingCount(nums, mid) >= k) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        //找到的是大于或者等于的情况
        return nums[left - 1] + k - missingCount(nums, left - 1);
    }

    private static int missingCount(int[] nums, int idx) {
        //应该有的数量 - 实际有的数量 = 缺失的数量
        return nums[idx] - nums[0] - (idx - 0);
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

    //[1095].山脉数组中查找目标值
    public class Solution1095 {
        public class MountainArray {
            public int get(int index) {
                return -1;
            }

            public int length() {
                return -1;
            }
        }

        public int findInMountainArray(int target, MountainArray mountainArr) {
            int len = mountainArr.length();
            int left = 0, right = len - 1;
            //找到顶峰位置
            while (left < right) {
                int mid = left + (right - left) / 2;
                if (mountainArr.get(mid) < mountainArr.get(mid + 1)) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }

            int index = leftBinarySearch(target, 0, left, mountainArr);
            if (index != -1) {
                return index;
            }
            return rightBinarySearch(target, left, len - 1, mountainArr);
        }

        private int leftBinarySearch(int target, int l, int r, MountainArray mountainArr) {
            while (l < r) {
                int mid = l + (r - l) / 2;
                if (mountainArr.get(mid) < target) {
                    l = mid + 1;
                } else {
                    r = mid;
                }
            }
            if (target == mountainArr.get(l)) {
                return l;
            }
            return -1;
        }

        private int rightBinarySearch(int target, int l, int r, MountainArray mountainArr) {
            while (l < r) {
                int mid = l + (r - l + 1) / 2;
                //递减区间找最左边的位置，排除掉小于的
                if (mountainArr.get(mid) < target) {
                    r = mid - 1;
                } else {
                    l = mid;
                }
            }
            if (target == mountainArr.get(l)) {
                return l;
            }
            return -1;
        }
    }

    //[1109].航班预订统计
    public int[] corpFlightBookings(int[][] bookings, int n) {
        Difference difference = new Difference(new int[n]);
        for (int[] booking : bookings) {
            difference.insert(booking[0] - 1, booking[1] - 1, booking[2]);
        }
        return difference.result();
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

    //[1155].掷骰子的N种方法
    public int numRollsToTarget(int n, int k, int target) {
        int mod = (int) 1e9 + 7;
        //物品维度n，背包维度target
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 0; i < n; i++) {
            for (int j = target; j >= 0; j--) {
                //三层遍历，需要置0操作
                dp[j] = 0;
                for (int x = 1; x <= k; x++) {
                    if (j >= x) {
                        dp[j] = (dp[j] + dp[j - x]) % mod;
                    }
                }
            }
        }
        return dp[target];
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

    //[1189].“气球” 的最大数量
    public int maxNumberOfBalloons(String text) {
        int[] cnt = new int[5];
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (ch == 'a') {
                cnt[0]++;
            } else if (ch == 'b') {
                cnt[1]++;
            } else if (ch == 'l') {
                cnt[2]++;
            } else if (ch == 'o') {
                cnt[3]++;
            } else if (ch == 'n') {
                cnt[4]++;
            }
        }
        int res = Integer.MAX_VALUE;
        for (int i = 0; i < cnt.length; i++) {
            if (i == 2 || i == 3) {
                res = Math.min(res, cnt[i] / 2);
            } else {
                res = Math.min(res, cnt[i]);
            }
        }
        return res == Integer.MAX_VALUE ? 0 : res;
    }

    //[1219].黄金矿工
    public int getMaximumGold(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        int max = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                visited[m][n] = true;
                AtomicInteger res = new AtomicInteger();
                dfsForGetMaximumGold(grid, i, j, visited, res);
                max = Math.max(max, res.get());
                visited[m][n] = false;
            }
        }
        return max;
    }

    private void dfsForGetMaximumGold(int[][] grid, int x, int y, boolean[][] visited, AtomicInteger count) {
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
        for (int[] direction : directions) {
            int nx = x + direction[0];
            int ny = y + direction[1];
            if (nx < 0 || ny < 0 || nx >= grid.length || ny >= grid[0].length
                    || visited[nx][ny] || grid[nx][ny] == 0) {
                continue;
            }

            visited[nx][ny] = true;
            count.addAndGet(grid[nx][ny]);
            dfsForGetMaximumGold(grid, nx, ny, visited, count);
            count.addAndGet(-grid[nx][ny]);
            visited[nx][ny] = false;
        }
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

    //[1233].删除子文件夹
    public List<String> removeSubfolders(String[] folder) {
        Arrays.sort(folder);
        List<String> res = new ArrayList<>();
        for (int i = 0; i < folder.length; ) {
            res.add(folder[i]);
            String dir = res.get(res.size() - 1);
            int j = i + 1;
            while (j < folder.length && folder[j].startsWith(dir + "/")) {
                j++;
            }
            i = j;
        }
        return res;
    }

    //[1245].树的直径
    int maxDiameter = 0;

    public int treeDiameter(int[][] edges) {
        Map<Integer, List<Integer>> g = new HashMap<>();
        for (int[] edge : edges) {
            g.putIfAbsent(edge[0], new ArrayList<>());
            g.putIfAbsent(edge[1], new ArrayList<>());
            g.get(edge[0]).add(edge[1]);
            g.get(edge[1]).add(edge[0]);
        }
        dfsForTreeDiameter(g, 0, -1);
        return maxDiameter;
    }

    private int dfsForTreeDiameter(Map<Integer, List<Integer>> g, int cur, int pre) {
        int max1 = 0, max2 = 0;
        for (int i : g.get(cur)) {
            if (i != pre) {
                int max = dfsForTreeDiameter(g, i, cur);
                if (max > max1) {
                    int temp = max1;
                    max1 = max;
                    max2 = temp;
                } else if (max > max2) {
                    max2 = max;
                }
            }
        }
        maxDiameter = Math.max(maxDiameter, max1 + max2);
        //最大的单边路径
        return max1 + 1;
    }

    //[1248].统计「优美子数组」
    public int numberOfSubarrays(int[] nums, int k) {
        //奇数个数，可以转化为奇数为1，偶数为0看待，求和为k的子数组个数
        Map<Integer, Integer> preSumCount = new HashMap<>();
        int res = 0, sum = 0;
        //代表和为0的时候，有一个计数，当k满足的时候，这个count就派上了用场。
        preSumCount.put(0, 1);
        for (int num : nums) {
            sum += num % 2 == 1 ? 1 : 0;
            if (preSumCount.containsKey(sum - k)) {
                res += preSumCount.get(sum - k);
            }
            preSumCount.put(sum, preSumCount.getOrDefault(sum, 0) + 1);
        }
        return res;
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

    //[1332].删除回文子序列
    public int removePalindromeSub(String s) {
        int n = s.length();
        int l = 0, r = n - 1;
        while (l < r) {
            if (s.charAt(l++) != s.charAt(r--)) return 2;
        }
        return 1;
    }

    //[1342].将数字变成 0 的操作次数
    public int numberOfSteps(int num) {
        return Math.max(getLoc(num) + getNum(num) - 1, 0);
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

    //[1360].日期之间隔几天
    public int daysBetweenDates(String date1, String date2) {
        return Math.abs(getDays(date1) - getDays(date2));
    }

    private int getDays(String date) {
        int[] MONTH_DAYS = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        String[] split = date.split("-");
        int year = Integer.parseInt(split[0]);
        int month = Integer.parseInt(split[1]);
        int day = Integer.parseInt(split[2]);
        int days = 0;
        for (int i = 1971; i < year; i++) {
            days += isleap(i) ? 366 : 365;
        }
        for (int i = 1; i < month; i++) {
            days += MONTH_DAYS[i];
        }
        if (isleap(year) && month > 2) {
            days++;
        }
        days += day;
        return days;
    }

    private boolean isleap(int year) {
        return year % 400 == 0 || (year % 4 == 0 && year % 100 != 0);
    }

    //[1363].形成三的最大倍数
    public static String largestMultipleOfThree(int[] digits) {
        //有对偶情况出现，两个2余为1，两个1余为2的情况
        //如果整体sum能被3整除，最大为本身的序
        //如果sum余3之后为1，优先删除最小的1，其次删除两个最小的2
        //如果sum余3之后为2，优先删除最小的2，其次删除两个最小的1
        //考虑桶排序情况，因为每个数字都是0~9，桶1,4,7为余数1， 桶2,5,8为余数2，拼接字符串时，从大到小排，根据每个桶的剩余个数排
        int sum = 0;
        int[] bucket = new int[10];
        for (int digit : digits) {
            bucket[digit]++;
            sum += digit;
        }
        sum %= 3;
        //剩下的个数
        int remain1 = bucket[1] + bucket[4] + bucket[7];
        int remain2 = bucket[2] + bucket[5] + bucket[8];
        //余数为1，优先删除1的个数，其次删除2的个数
        if (sum == 1) {
            if (remain1 > 0) {
                remain1--;
            } else {
                remain2 -= 2;
            }
        } else if (sum == 2) {
            //余数为1，优先删除2的个数，其次删除1的个数
            if (remain2 > 0) {
                remain2--;
            } else {
                remain1 -= 2;
            }
        }
        StringBuilder sb = new StringBuilder();
        //每个数字拼接的时候，从大到小，这样删除的数永远是最小的
        for (int num = 9; num >= 0; num--) {
            if (num % 3 == 1) {
                while (remain1 > 0 && bucket[num] > 0) {
                    remain1--;
                    sb.append(num);
                    bucket[num]--;
                }
            } else if (num % 3 == 2) {
                while (remain2 > 0 && bucket[num] > 0) {
                    remain2--;
                    sb.append(num);
                    bucket[num]--;
                }
            } else {
                while (bucket[num] > 0) {
                    sb.append(num);
                    bucket[num]--;
                }
            }
        }
        return sb.length() > 0 && sb.charAt(0) == '0' ? "0" : sb.toString();
    }

    //[1373].二叉搜索子树的最大键值和
    public int maxSumBST(TreeNode root) {
        //后序遍历，父节点依赖子节点计算逻辑
        AtomicInteger maxSum = new AtomicInteger(Integer.MIN_VALUE);
        dfsForMaxSumBST(root, maxSum);
        return maxSum.get();
    }

    private int[] dfsForMaxSumBST(TreeNode root, AtomicInteger maxSum) {
        //是否是二叉搜索树，最小值，最大值，和
        if (root == null) return new int[]{1, Integer.MAX_VALUE, Integer.MIN_VALUE, 0};

        int[] left = dfsForMaxSumBST(root.left, maxSum);
        int[] right = dfsForMaxSumBST(root.right, maxSum);

        int[] res = new int[4];
        //根节点小于右侧的最小值，大于左侧的最大值
        if (left[0] == 1 && right[0] == 1 && root.val < right[1] && root.val > left[2]) {
            //统计和
            res[0] = 1;
            res[1] = Math.min(left[1], root.val);
            res[2] = Math.max(right[2], root.val);
            res[3] = left[3] + right[3] + root.val;
            maxSum.set(Math.max(maxSum.get(), res[3]));
        } else {
            res[0] = 0;
        }
        return res;
    }

    private int getLoc(int num) {
        for (int i = 31; i >= 0; i--) {
            if (((num >> i) & 1) == 1) return i + 1;
        }
        return -1;
    }

    private int getNum(int num) {
        int ans = 0;
        for (int i = 31; i >= 0; i--) {
            if (((num >> i) & 1) == 1) {
                ans++;
            }
        }
        return ans;
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

    //[1380].矩阵中的幸运数
    public List<Integer> luckyNumbers(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[] row = new int[m];
        int[] col = new int[n];
        for (int i = 0; i < m; i++) {
            row[i] = Integer.MAX_VALUE;
            for (int j = 0; j < n; j++) {
                row[i] = Math.min(matrix[i][j], row[i]);
                col[j] = Math.max(matrix[i][j], col[j]);
            }
        }

        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == row[i] && matrix[i][j] == col[j]) {
                    res.add(matrix[i][j]);
                }
            }
        }
        return res;
    }

    //[1405].最长快乐字符串
    public static String longestDiverseString(int a, int b, int c) {
        PriorityQueue<int[]> queue = new PriorityQueue<>((x, y) -> y[1] - x[1]);
        if (a > 0) queue.offer(new int[]{0, a});
        if (b > 0) queue.offer(new int[]{1, b});
        if (c > 0) queue.offer(new int[]{2, c});
        StringBuilder sb = new StringBuilder();
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int n = sb.length();
            //之前拼接的个数为2，则换下一个，值没超过2，继续拼接
            if (n >= 2 && sb.charAt(n - 1) - 'a' == cur[0] && sb.charAt(n - 2) - 'a' == cur[0]) {
                if (queue.isEmpty()) break;
                int[] next = queue.poll();

                sb.append((char) (next[0] + 'a'));
                if (--next[1] > 0) {
                    queue.offer(next);
                }
                queue.offer(cur);
            } else {
                sb.append((char) (cur[0] + 'a'));
                if (--cur[1] > 0) {
                    queue.offer(cur);
                }
            }
        }
        return sb.toString();
    }

    //[1414].和为 K 的最少斐波那契数字数目
    public int findMinFibonacciNumbers(int k) {
        List<Integer> f = new ArrayList<>();
        f.add(1);
        int a = 1, b = 1;
        while (a + b <= k) {
            int c = a + b;
            f.add(c);
            a = b;
            b = c;
        }
        int ans = 0;
        for (int i = f.size() - 1; i >= 0 && k > 0; i--) {
            int num = f.get(i);
            if (k >= num) {
                k -= num;
                ans++;
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

    //[1447].最简分数
    public List<String> simplifiedFractions(int n) {
        List<String> res = new ArrayList<>();
        for (int i = 1; i < n; i++) {
            for (int j = i + 1; j <= n; j++) {
                if (gcd2(i, j) == 1) {
                    res.add(i + "/" + j);
                }
            }
        }
        return res;
    }

    private int gcd2(int a, int b) {
        return b == 0 ? a : gcd2(b, a % b);
    }

    //[1458].两个子序列的最大点积
    public int maxDotProduct(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        //以i为结尾的和以j为结尾的两个子序列的最大点积
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int xy = nums1[i] * nums2[j];
                dp[i][j] = xy;
                if (i > 0) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - 1][j]);
                }
                if (j > 0) {
                    dp[i][j] = Math.max(dp[i][j], dp[i][j - 1]);
                }
                if (i > 0 && j > 0) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - 1][j - 1] + xy);
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    //[1493].删掉一个元素以后全为 1 的最长子数组
    public int longestSubarray(int[] nums) {
        //当窗口内的0的元素个数> 1的时候需要缩窗口，否则扩窗口
        //0,1,1,1,0,1,1,0,1
        int n = nums.length, zeroCount = 0, res = 0;
        for (int l = 0, r = 0; r < n; r++) {
            if (nums[r] == 0) zeroCount++;

            while (zeroCount > 1) {
                if (nums[l] == 0) {
                    zeroCount--;
                }
                l++;
            }
            res = Math.max(res, r - l + 1 - zeroCount);
        }
        return res == n ? n - 1 : res;
    }

    //[1497].检查数组对是否可以被 k 整除
    public boolean canArrange(int[] arr, int k) {
        int[] mod = new int[k];
        for (int num : arr) {
            //把负数的余数调整成[0, k-1]中间
            ++mod[(num % k + k) % k];
        }
        //余数1开始判断配对的数，余数和为k的数量必须相等
        for (int i = 1; i + i < k; ++i) {
            if (mod[i] != mod[k - i]) {
                return false;
            }
        }
        //余数0，需要偶数对。
        return mod[0] % 2 == 0;
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

    //[1601].最多可达成的换楼请求数目
    public int maximumRequests(int n, int[][] requests) {
        int m = requests.length;
        int ans = 0;
        //请求的状态2^m-1种选择
        for (int i = 0; i < (1 << m); i++) {
            int cnt = getCnt(i);
            if (cnt <= ans) continue;
            if (check(i, requests)) ans = cnt;
        }
        return ans;
    }

    private boolean check(int s, int[][] requests) {
        //每种状态使得每个楼的数据都是净变化为 0
        int[] cnt = new int[20];
        int sum = 0;
        for (int j = 0; j < 16; j++) {
            //一共16个状态位，计算判断每一个请求导致的变化
            if (((s >> j) & 1) == 1) {
                if (++cnt[requests[j][0]] == 1) sum++;
                if (--cnt[requests[j][1]] == 0) sum--;
            }
        }
        return sum == 0;
    }

    private int getCnt(int s) {
        int ans = 0;
        for (int i = s; i > 0; i -= (i & -i)) {
            ans++;
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

    //[1642].可以到达的最远建筑
    public int furthestBuilding(int[] heights, int bricks, int ladders) {
        PriorityQueue<Integer> queue = new PriorityQueue<>();
        for (int i = 1; i < heights.length; i++) {
            int cnt = heights[i] - heights[i - 1];
            if (cnt <= 0) {
                continue;
            }
            queue.offer(cnt);
            if (queue.size() > ladders) {
                //最小的砖头替代
                bricks -= queue.poll();
            }
            if (bricks < 0) {
                return i - 1;
            }
        }
        return heights.length - 1;
    }


    //[1654].到家的最少跳跃次数
    public int minimumJumps(int[] forbidden, int a, int b, int x) {
        boolean[] isForbidden = new boolean[6001];
        for (int i : forbidden) isForbidden[i] = true;
        Queue<int[]> queue = new PriorityQueue<>(Comparator.comparingInt(o -> o[1]));
        queue.offer(new int[]{0, 0});
        boolean[] visited = new boolean[6001];
        while (!queue.isEmpty()) {
            int[] o = queue.poll();
            int curr = o[0], step = o[1];
            if (curr == x) return step;
            if (visited[curr]) continue;
            visited[curr] = true;
            // 前进一步
            int next = curr + a;
            if (next > 6000 || isForbidden[next]) continue;
            queue.offer(new int[]{next, step + 1});
            // 前进一步+后退一步，连走两步
            next -= b;
            if (next < 0 || isForbidden[next]) continue;
            queue.offer(new int[]{next, step + 2});
        }
        return -1;
    }

    //[1688].比赛中的配对次数
    public int numberOfMatches(int n) {
        return n - 1;
    }

    //[1690].石子游戏 VII
    public int stoneGameVII(int[] stones) {
        int n = stones.length;
        //同样定义为i到j的范围内，先手的最大分差值
        int[][] dp = new int[n][n];
        //方便求解区间的sum
        int[][] sum = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (j == i) {
                    sum[i][j] = stones[j];
                } else {
                    sum[i][j] = sum[i][j - 1] + stones[j];
                }
            }
        }
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (i == j) {
                    //只有一个元素可选时，没有利益
                    dp[i][j] = 0;
                } else if (i == j - 1) {
                    //有两个元素时，一定选择使得剩下值最大的另一个
                    dp[i][j] = Math.max(stones[i], stones[j]);
                } else {
                    //先手选择i元素，剩下的和是他的利益 与 后手选择的最大利益之间的差值
                    int left = sum[i + 1][j] - dp[i + 1][j];
                    //先手选择j元素，剩下的和是他的利益 与 后手选择的最大利益之间的差值
                    int right = sum[i][j - 1] - dp[i][j - 1];
                    //先手选择一个最大利益值
                    dp[i][j] = Math.max(left, right);
                }
            }
        }
        return dp[0][n - 1];
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

    //[1706].球会落何处
    public int[] findBall(int[][] grid) {
        int n = grid[0].length;
        int[] ans = new int[n];
        for (int j = 0; j < n; j++) {
            int col = j;
            for (int[] row : grid) {
                int dir = row[col];
                col += dir;
                //水平方向上能形成V的肯定
                if (col < 0 || col >= n || row[col] != dir) {
                    col = -1;
                    break;
                }
            }
            ans[j] = col;
        }
        return ans;
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

    //[1725].可以形成最大正方形的矩形数目
    public int countGoodRectangles(int[][] rectangles) {
        int count = 0, maxSide = 0;
        int n = rectangles.length;
        for (int[] rectangle : rectangles) {
            int minSide = Math.min(rectangle[0], rectangle[1]);
            if (minSide == maxSide) {
                count++;
            } else if (minSide > maxSide) {
                count = 1;
                maxSide = minSide;
            }
        }
        return count;
    }

    //[1763].最长的美好子字符串
    public String longestNiceSubstring(String s) {
        int n = s.length();
        int count = 0;
        String ans = "";
        for (int i = 0; i < n; i++) {
            int lower = 0, upper = 0;
            //不关心字符的数量，只关心有没有
            for (int j = i; j < n; j++) {
                char ch = s.charAt(j);
                if (Character.isLowerCase(ch)) {
                    lower |= 1 << (s.charAt(j) - 'a');
                } else {
                    upper |= 1 << (s.charAt(j) - 'A');
                }
                if (lower == upper && j - i + 1 > count) {
                    ans = s.substring(i, j + 1);
                    count = j - i + 1;
                }
            }
        }
        return ans;
    }

    //[1765].地图中的最高点
    public int[][] highestPeak(int[][] isWater) {
        //广度优先遍历的特点是层，可以解决最短路径，树层遍历，图遍历
        //该道题需要从水域开始遍历，往外一层层遍历，后面节点重复遍历必然值更大，但不满足题目要求
        int m = isWater.length, n = isWater[0].length;
        int[][] res = new int[m][n];
        Queue<int[]> queue = new ArrayDeque<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (isWater[i][j] == 1) {
                    queue.offer(new int[]{i, j});
                }
                res[i][j] = isWater[i][j] == 1 ? 0 : -1;
            }
        }
        int[][] directs = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0], y = cur[1];
            int val = res[x][y];
            for (int[] dir : directs) {
                int nx = dir[0] + x;
                int ny = dir[1] + y;
                if (nx < 0 || ny < 0 || nx >= m || ny >= n || res[nx][ny] != -1) {
                    continue;
                }
                res[nx][ny] = val + 1;
                queue.offer(new int[]{nx, ny});
            }
        }
        return res;
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

    //[1823].找出游戏的获胜者
    public int findTheWinner(int n, int k) {
        if (n <= 1) return n;
        int ans = (findTheWinner(n - 1, k) + k) % n;
        return ans == 0 ? n : ans;
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

    //[1984].学生分数的最小差值
    public int minimumDifference(int[] nums, int k) {
        Arrays.sort(nums);
        int n = nums.length;
        int ans = nums[k - 1] - nums[0];
        for (int i = 1; i <= n - k; i++) {
            ans = Math.min(nums[i + k - 1] - nums[i], ans);
        }
        return ans;
    }

    //[1994].好子集的数目
    public int numberOfGoodSubsets(int[] nums) {
        int MOD = (int) 1e9 + 7;
        // 30 以内的质数
        int[] p = new int[]{2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
        // 统计数字出现的次数
        int[] cnts = new int[31];
        for (int num : nums) cnts[num]++;

        int mask = 1 << 10;
        //dp[5]  5 二进制位 0000000101, 表示子集所有元素乘积为 p[0] * p[2] = 2 * 5 的个数
        // 动态转移方式为在已有的情况下再增加其他质数  (p[0] * p[2]) * p[1] = (2 * 5) * 3
        // 注意这里 dp 用 int 会溢出
        long[] f = new long[mask];
        f[0] = 1;
        // 尝试将 num 加入子集中，并累计结果数
        for (int num = 2; num <= 30; num++) {
            if (cnts[num] == 0) continue;
            boolean ok = true;
            int t = num, subset = 0;
            // t 用来你分解质因数， mask 二进制位用来记录分解出现的质因数位置
            for (int j = 0; j < p.length; j++) {
                int cnt = 0;
                while ((t % p[j]) == 0) {
                    subset |= (1 << j);
                    t /= p[j];
                    cnt++;
                }
                if (cnt > 1) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            for (int prev = mask - 1; prev >= 0; prev--) {
                //有交集：相同质数
                if ((prev & subset) != 0) continue;
                //不选择i作为子集，选择i作为子集
                f[prev | subset] = (f[prev | subset] + f[prev] * cnts[num]) % MOD;
            }
        }
        long ans = 0;
        //统计所有状态的方案数
        for (int i = 1; i < mask; i++) ans = (ans + f[i]) % MOD;
        // 在此基础上，考虑每个 1 选择与否对答案的影响
        for (int i = 0; i < cnts[1]; i++) ans = ans * 2 % MOD;
        return (int) ans;
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

    //[1996].游戏中弱角色的数量
    public int numberOfWeakCharacters(int[][] properties) {//第一维度降序，第二维度增序，保证相同的攻击值的时候，最大的防御值大，一定是一个弱者
        Arrays.sort(properties, (a, b) -> a[0] == b[0] ? a[1] - b[1] : b[0] - a[0]);

        int maxDef = 0, ans = 0;
        for (int[] property : properties) {
            if (property[1] < maxDef) {
                ans++;
            } else {
                maxDef = property[1];
            }
        }
        return ans;
    }

    //[2000].反转单词前缀
    public String reversePrefix(String word, char ch) {
        char[] arr = word.toCharArray();
        int i = 0;
        while (i < arr.length) {
            if (ch == arr[i]) break;
            i++;
        }
        if (i == arr.length) return word;

        int l = 0, r = i;
        while (l < r) {
            char temp = arr[l];
            arr[l] = arr[r];
            arr[r] = temp;
            l++;
            r--;
        }
        return String.valueOf(arr);
    }

    //[2006].差的绝对值为 K 的数对数目
    public int countKDifference(int[] nums, int k) {
        int ans = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            ans += map.getOrDefault(num - k, 0);
            ans += map.getOrDefault(num + k, 0);

            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        return ans;
    }

    //[2013].检测正方形
    public class DetectSquares {

        Map<Integer, Map<Integer, Integer>> row2Col;

        public DetectSquares() {
            row2Col = new HashMap<>();
        }

        public void add(int[] point) {
            int x = point[0], y = point[1];
            Map<Integer, Integer> yCount = row2Col.getOrDefault(x, new HashMap<>());
            yCount.put(y, yCount.getOrDefault(y, 0) + 1);
            row2Col.put(x, yCount);
        }

        public int count(int[] point) {
            int x = point[0], y = point[1];
            int ans = 0;
            Map<Integer, Integer> yCount = row2Col.getOrDefault(x, new HashMap<>());
            //上下不同的点
            for (int ny : yCount.keySet()) {
                if (ny == y) continue;
                //正上方
                int c1 = yCount.get(ny);
                int d = ny - y;
                //左右不同方向
                int[] xs = new int[]{x + d, x - d};
                for (int nx : xs) {
                    Map<Integer, Integer> nyCount = row2Col.getOrDefault(nx, new HashMap<>());
                    //左上方
                    int c3 = nyCount.getOrDefault(ny, 0);
                    //左下方
                    int c2 = nyCount.getOrDefault(y, 0);
                    ans += c1 * c2 * c3;
                }
            }
            return ans;
        }
    }

    //[2016].增量元素之间的最大差值
    public int maximumDifference(int[] nums) {
        int min = nums[0], ans = -1;
        for (int i = 1; i < nums.length; i++) {
            if (min < nums[i]) {
                ans = Math.max(ans, nums[i] - min);
            } else {
                min = nums[i];
            }
        }
        return ans;
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

    //[2024].考试的最大困扰度
    public int maxConsecutiveAnswers(String answerKey, int k) {
        int countT = 0, countF = 0;
        int ans = 0;
        for (int l = 0, r = 0; r < answerKey.length(); r++) {
            char ch = answerKey.charAt(r);
            if (ch == 'T') countT++;
            else countF++;

            //这个条件没找到，应该是最小值超过了k，意味着需要缩窗口
            while (Math.min(countF, countT) > k && l < r) {
                char lCh = answerKey.charAt(l);
                if (lCh == 'T') countT--;
                else countF--;
                l++;
            }

            ans = Math.max(ans, r - l + 1);
        }
        return ans;
    }

    //[2028].找出缺失的观测数据
    public int[] missingRolls(int[] rolls, int mean, int n) {
        int m = rolls.length;
        int remain = mean * (m + n);
        for (int roll : rolls) {
            remain -= roll;
        }
        if (remain < n || remain > 6 * n) return new int[0];
        int avg = remain / n, remainder = remain % n;
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            ans[i] = avg + (i < remainder ? 1 : 0);
        }
        return ans;
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

    //[2038].如果相邻两个颜色均相同则删除当前颜色
    public boolean winnerOfGame(String colors) {
        int n = colors.length();
        int a = 0, b = 0;
        for (int i = 1; i < n - 1; i++) {
            if (colors.charAt(i - 1) == 'A' && colors.charAt(i) == 'A' && colors.charAt(i + 1) == 'A') {
                a++;
            }
            if (colors.charAt(i - 1) == 'B' && colors.charAt(i) == 'B' && colors.charAt(i + 1) == 'B') {
                b++;
            }
        }
        return a > b;
    }

    //[2044].统计按位或能得到最大值的子集数目
    public int countMaxOrSubsets(int[] nums) {
        //不能通过32位来统计每位1的个数相乘，因为有可能是某一位都是0

//        int ans = 0, max = 0;
//        //可以穷举所有子集的状态，来计算最大值和子集数目
//        int n = nums.length, state = 1 << n;
//        for (int s = 0; s < state; s++) {
//
//            int cur = 0;
//            //遍历每个数，看看子集是不是选中
//            for (int i = 0; i < n; i++) {
//                if ((s >> i & 1) == 1) {
//                    cur |= nums[i];
//                }
//            }
//            if (cur > max) {
//                max = cur;
//                ans = 1;
//            } else if (cur == max) {
//                ans++;
//            }
//        }
//        return ans;

        AtomicInteger max = new AtomicInteger();
        AtomicInteger ans = new AtomicInteger();
        dfsForCountMaxOrSubsets(0, 0, nums, max, ans);
        return ans.get();
    }

    private void dfsForCountMaxOrSubsets(int start, int val, int[] nums, AtomicInteger max, AtomicInteger ans) {
        if (start == nums.length) {
            if (val > max.get()) {
                max.set(val);
                ans.set(1);
            } else if (val == max.get()) {
                ans.incrementAndGet();
            }
            return;
        }

        //不选择
        dfsForCountMaxOrSubsets(start + 1, val, nums, max, ans);
        //选择
        dfsForCountMaxOrSubsets(start + 1, val | nums[start], nums, max, ans);
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

    //[2047].句子中的有效单词数
    public int countValidWords(String sentence) {
        int ans = 0, n = sentence.length();
        for (int i = 0; i < n; ) {
            if (sentence.charAt(i) == ' ') continue;
            int j = i;
            while (j < n && sentence.charAt(j) != ' ') j++;
            if (checkForCountValidWords(sentence.substring(i, j))) {
                ans++;
            }
            i = j + 1;
        }
        return ans;
    }

    private boolean checkForCountValidWords(String word) {
        int countDash = 0, countSign = 0;
        if (word.length() == 0) return false;
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            if (ch >= '0' && ch <= '9') {
                return false;
            } else if (ch == ' ') {
                return false;
            } else if (ch == '-') {
                countDash++;
                if (i == 0 || i == word.length() - 1
                        || !Character.isLetter(word.charAt(i - 1))
                        || !Character.isLetter(word.charAt(i + 1))
                        || countDash > 1) {
                    return false;
                }
            } else if (ch == '!' || ch == '.' || ch == ',') {
                countSign++;
                if (countSign > 1 || i != word.length() - 1) {
                    return false;
                }
            }
        }
        return true;
    }

    //[2049]统计最高分的节点数目
    public class Solution2049 {
        int ans = 0;
        long maxScore = 0;

        public int countHighestScoreNodes(int[] parents) {
            int n = parents.length;
            int[] left = new int[n];
            int[] right = new int[n];
            Arrays.fill(left, -1);
            Arrays.fill(right, -1);
            //0的父亲节点是-1，不需要建
            for (int i = 1; i < n; i++) {
                int parent = parents[i];
                if (left[parent] == -1) {
                    left[parent] = i;
                } else {
                    right[parent] = i;
                }
            }
            dfs(0, left, right, n);

            return ans;
        }

        private int dfs(int node, int[] left, int[] right, int n) {
            if (node == -1) {
                return 0;
            }

            int leftCount = dfs(left[node], left, right, n);
            int rightCount = dfs(right[node], left, right, n);
            //除去本身这棵树的数量
            int remain = n - leftCount - rightCount - 1;
            long score = count(leftCount) * count(rightCount) * count(remain);

            if (score == maxScore) {
                ans++;
            } else if (score > maxScore) {
                maxScore = score;
                ans = 1;
            }

            return leftCount + rightCount + 1;
        }

        private long count(int count) {
            return count == 0 ? 1 : count;
        }
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

    //[2100].适合打劫银行的日子
    public List<Integer> goodDaysToRobBank(int[] security, int time) {
        int n = security.length;
        //i的左边，连续非递增的天数
        int[] left = new int[n];
        //i的右边，连续非递减的天数
        int[] right = new int[n];
        for (int i = 1; i < n; i++) {
            if (security[i - 1] >= security[i]) {
                left[i] = left[i - 1] + 1;
            }

            if (security[n - i - 1] <= security[n - i]) {
                right[n - i - 1] = right[n - i] + 1;
            }
        }
        List<Integer> res = new ArrayList<>();
        for (int i = time; i < n - time; i++) {
            if (left[i] >= time && right[i] >= time) {
                res.add(i);
            }
        }
        return res;
    }

    //[2108].找出数组中的第一个回文字符串
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

    //[2109].向字符串添加空格
    public static String addSpaces(String s, int[] spaces) {
        StringBuilder sb = new StringBuilder();
        int start = 0;
        for (int space : spaces) {
            sb.append(s.substring(start, space));
            sb.append(' ');
            start = space;
        }
        sb.append(s.substring(start));
        return sb.toString();
    }

    //[2110].股票平滑下跌阶段的数目
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

    //[2111].使数组 K 递增的最少操作次数
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

    //[2119].反转两次的数字
    public boolean isSameAfterReversals(int num) {
        return num > 0 ? num % 10 != 0 : true;
    }

    //[2120].执行所有后缀指令
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

    //[2121].相同元素的间隔之和
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

    //[2133].检查是否每一行每一列都包含全部整数
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

    //[2134].最少交换次数来组合所有的 1 II
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

    //[2135].统计追加字母可以获得的单词数
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

    //[2138].将字符串拆分为若干长度为 k 的组
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

    //[2139].得到目标值的最少行动次数
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

    //[2140].解决智力问题
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

    //[2148].元素计数
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

    //[2149].按符号重排数组
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

    //[2150].找出数组中的所有孤独数字
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

    //[2169].得到 0 的操作数
    public static int countOperations(int num1, int num2) {
        int count = 0;
        while (num1 != 0 && num2 != 0) {
            if (num1 > num2) {
                count++;
                num1 = num1 - num2;
            } else if (num1 < num2) {
                count++;
                num2 = num2 - num1;
            } else if (num1 == num2) {
                count++;
                num2 = num2 - num1;
            }
        }
        return count;
    }

    //[2170].使数组变成交替数组的最少操作数
    public static int minimumOperations(int[] nums) {
        Map<Integer, Integer> count = new HashMap<>();
        int n = nums.length;
        int maxEvenCount = 0, secondEvenCount = 0, maxEven = -1;
        for (int i = 0; i < n; i += 2) {
            count.put(nums[i], count.getOrDefault(nums[i], 0) + 1);
        }
        for (int key : count.keySet()) {
            if (count.get(key) > maxEvenCount) {
                secondEvenCount = maxEvenCount;
                maxEvenCount = count.get(key);
                maxEven = key;
            } else if (count.get(key) > secondEvenCount) {
                secondEvenCount = count.get(key);
            }
        }

        count = new HashMap<>();
        int maxOddCount = 0, secondOddCount = 0, maxOdd = -1;
        for (int i = 1; i < n; i += 2) {
            count.put(nums[i], count.getOrDefault(nums[i], 0) + 1);
        }
        for (int key : count.keySet()) {
            if (count.get(key) > maxOddCount) {
                secondOddCount = maxOddCount;
                maxOddCount = count.get(key);
                maxOdd = key;
            } else if (count.get(key) > secondOddCount) {
                secondOddCount = count.get(key);
            }
        }

        if (maxOdd != maxEven) {
            //odd是保留数量， even是保留数量，剩下的就是变更最小数量
            return n - maxOddCount - maxEvenCount;
        }
        //如果相等，需要去一个最大的保留数量
        //去掉奇数最大之后，保留第二大的+偶数的数量，或者去掉偶数最大的，保留第二大+奇数的数量
        int max = Math.max(secondOddCount + maxEvenCount, secondEvenCount + maxOddCount);
        return n - max;
    }

    //[2171].拿出最少数目的魔法豆
    public static long minimumRemoval(int[] beans) {
        int n = beans.length;
        Arrays.sort(beans);
        long sum = 0;
        for (int bean : beans) {
            sum += bean;
        }
        long min = sum;
        for (int i = 0; i < n; i++) {
            long cnt = n - i;
            long bean = beans[i];
            long left = cnt * bean;
            min = Math.min(min, sum - left);
        }
        return min;
    }

    //[2180].统计各位数字之和为偶数的整数个数
    public int countEven(int num) {
        int cur = 1, ans = 0;
        while (cur <= num) {
            if (isValidCountEven(cur)) {
                ans++;
            }
            cur++;
        }
        return ans;
    }

    private boolean isValidCountEven(int num) {
        int sum = 0;
        while (num != 0) {
            sum += num % 10;
            num /= 10;
        }
        return (sum % 2) == 0;
    }

    //[2181].合并零之间的节点
    public ListNode mergeNodes(ListNode head) {
        ListNode dummy = new ListNode(-1), tail = dummy;
        int sum = 0;
        while (head != null) {
            if (head.val == 0) {
                if (sum != 0) {
                    tail.next = new ListNode(sum);
                    tail = tail.next;
                }
                sum = 0;
            } else {
                sum += head.val;
            }
            head = head.next;
        }
        return dummy.next;
    }

    //[2182].构造限制重复的字符串
    public String repeatLimitedString(String s, int repeatLimit) {
        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> b[0] - a[0]);
        int[] cnt = new int[26];
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            cnt[ch - 'a']++;
        }
        for (int a = 0; a < 26; a++) {
            if (cnt[a] > 0) {
                queue.offer(new int[]{a, cnt[a]});
            }
        }
        StringBuilder sb = new StringBuilder();
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            if (cur[1] > repeatLimit) {
                int size = repeatLimit;
                while (size-- > 0) sb.append((char) (cur[0] + 'a'));
                if (queue.isEmpty()) return sb.toString();
                int[] next = queue.peek();
                if (next[1] == 1) {
                    queue.poll();
                } else {
                    next[1]--;
                }
                sb.append((char) (next[0] + 'a'));
                queue.offer(new int[]{cur[0], cur[1] - repeatLimit});
            } else {
                int size = cur[1];
                while (size-- > 0) sb.append((char) (cur[0] + 'a'));
            }
        }
        return sb.toString();
    }

    //[2183].统计可以被 K 整除的下标对数目
    public long countPairs(int[] nums, int k) {
        //(a * b) % k => (gcd(a, k) * gcd(b, k)) % k
        //nums [2, 9]   K = 6
        //不同的gcd之间的数量数乘积，相同gcd，也有可能被k整除，还需要单独计算。
        Map<Integer, Long> cnt = new HashMap<>();
        for (int num : nums) {
            //求与k的最大公约数
            int gcd = gcd(k, num);
            cnt.put(gcd, cnt.getOrDefault(gcd, 0L) + 1L);
        }

        long sum = 0;
        for (int k1 : cnt.keySet()) {
            for (int k2 : cnt.keySet()) {
                int t = (k1 * k2) % k;
                //防止重复计算，只统计有序的
                if (k1 < k2 && t == 0) {
                    sum += cnt.get(k1) * cnt.get(k2);
                } else if (k1 == k2 && t == 0) {
                    //相等的k，也可能导致能够被k整除
                    long count = cnt.get(k1);
                    sum += count * (count - 1) / 2;
                }
            }
        }
        return sum;
    }

    //[6008].统计包含给定前缀的字符串
    public int prefixCount(String[] words, String pref) {
        int ans = 0;
        for (String word : words) {
            if (word.startsWith(pref)) {
                ans++;
            }
        }
        return ans;
    }

    //[6009].使两字符串互为字母异位词的最少步骤数
    public int minSteps(String s, String t) {
        int[] cnt1 = new int[26];
        for (int i = 0; i < s.length(); i++) {
            int ch = s.charAt(i) - 'a';
            cnt1[ch]++;
        }

        int[] cnt2 = new int[26];
        for (int i = 0; i < t.length(); i++) {
            int ch = t.charAt(i) - 'a';
            cnt2[ch]++;
        }
        int ans = 0;
        for (int i = 0; i < 26; i++) {
            ans += Math.abs(cnt1[i] - cnt2[i]);
        }
        return ans;
    }

    //[6010].完成旅途的最少时间
    public long minimumTime(int[] time, int totalTrips) {
        Arrays.sort(time);
        long right = (long) totalTrips * time[0];
        long left = 1;
        while (left < right) {
            long mid = left + (right - left) / 2;
            long trips = 0;
            for (int i = 0; i < time.length; i++) {
                trips += mid / time[i];
            }

            if (trips >= totalTrips) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    //[6037].按奇偶性交换后的最大数字
    public int largestInteger(int num) {
        PriorityQueue<Integer> even = new PriorityQueue<>((a, b) -> b - a);
        PriorityQueue<Integer> odd = new PriorityQueue<>((a, b) -> b - a);

        for (char ch : (num + "").toCharArray()) {
            (ch % 2 > 0 ? odd : even).offer(ch - '0');
        }

        int res = 0;
        for (char ch : (num + "").toCharArray()) {
            res = res * 10 + (ch % 2 > 0 ? odd : even).poll();
        }
        return res;
    }

    //[6038].向表达式添加括号后的最小结果
    public String minimizeResult(String expression) {
        String[] split = expression.split("\\+");
        int min = Integer.MAX_VALUE;
        String ans = "";
        for (int i = 0; i < split[0].length(); i++) {
            for (int j = 1; j <= split[1].length(); j++) {
                int cur = (i > 0 ? Integer.parseInt(split[0].substring(0, i)) : 1) *
                        (Integer.parseInt(split[0].substring(i)) + Integer.parseInt(split[1].substring(0, j))) *
                        (j == split[1].length() ? 1 : Integer.parseInt(split[1].substring(j)));
                if (cur < min) {
                    min = cur;
                    ans = split[0].substring(0, i)
                            + '(' + split[0].substring(i) + '+' + split[1].substring(0, j) + ')'
                            + split[1].substring(j);
                }
            }
        }
        return ans;
    }

    //[6039].K 次增加后的最大乘积
    public static int maximumProduct(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        int BASE = (int) 1e9 + 7;
        for (int num : nums) {
            pq.offer(num);
        }
        while (!pq.isEmpty() && k > 0) {
            pq.offer(pq.poll() + 1);
            k--;
        }
        long ans = 1;
        while (!pq.isEmpty()) {
            ans *= pq.poll();
            ans %= BASE;
        }
        return (int) ans;
    }


    //[剑指 Offer 03].数组中重复的数字
    public int findRepeatNumber(int[] nums) {
        //原地hash算法，已知数字范围是从0到n-1，将数字与实际的值交换
        //[2, 3, 1, 0, 2, 5, 3]
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            //如果当前位置上的索引和当前值不相等；如果相等继续下一个
            while (nums[i] != i) {

                //重复的数字必然会跟正确的位置闹冲突
                if (nums[nums[i]] == nums[i]) {
                    return nums[i];
                }

                //将当前值映射到正确的索引位置上
                int temp = nums[nums[i]];
                nums[nums[i]] = nums[i];
                nums[i] = temp;
            }
        }
        return -1;
    }

    //[剑指 Offer 06].从尾到头打印链表
    private List<Integer> ans = new ArrayList<>();

    public int[] reversePrint(ListNode head) {
        helperForReversePrint(head);
        int[] res = new int[ans.size()];
        for (int i = 0; i < res.length; i++) {
            res[i] = ans.get(i);
        }
        return res;
    }

    private void helperForReversePrint(ListNode head) {
        if (head == null) return;
        helperForReversePrint(head.next);
        ans.add(head.val);
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

    //[剑指 Offer 13].机器人的运动范围
    public int movingCount(int m, int n, int k) {
        if (k == 0) return 1;
        boolean[][] visited = new boolean[m][n];
        return dfsForMovingCount(0, 0, k, m, n, visited);
    }

    private int dfsForMovingCount(int x, int y, int k, int m, int n, boolean[][] visited) {
        if (x < 0 || y < 0 || x >= m || y >= n || visited[x][y]) return 0;
        int[][] directs = new int[][]{{0, 1}, {1, 0}};
        int res = 0;
        visited[x][y] = true;
        for (int[] dir : directs) {
            int nx = x + dir[0];
            int ny = y + dir[1];
            if (!checkValidForMovingCount(nx, ny, k)) {
                continue;
            }
            res += dfsForMovingCount(nx, ny, k, m, n, visited);
        }
        return res + 1;
    }

    private boolean checkValidForMovingCount(int x, int y, int k) {
        int sum = 0;
        while (x > 0) {
            sum += x % 10;
            x = x / 10;
        }
        while (y > 0) {
            sum += y % 10;
            y = y / 10;
        }
        return sum <= k;
    }

    //[剑指 Offer 21].调整数组顺序使奇数位于偶数前面
    public int[] exchange(int[] nums) {
//        int fast = 0, slow = 0;
//        //fast定位奇数，slow为下一个需要替换的奇数位
//        while (fast < nums.length) {
//            if ((nums[fast] & 1) == 1) {
//                int temp = nums[slow];
//                nums[slow] = nums[fast];
//                nums[fast] = temp;
//                slow++;
//            }
//            fast++;
//        }
//        return nums;
        //快速交换，就是一次划分，分割两边
        //保证i的左侧都是奇数，j的右侧都是偶数
        int i = 0, j = nums.length - 1;
        while (i < j) {
            //直到找到一个偶数
            while (i < j && (nums[i] & 1) == 1) i++;
            //直到找到一个奇数
            while (i < j && (nums[j] & 1) == 0) j--;

            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
        }
        return nums;
    }

    //[剑指 Offer 30].包含min函数的栈
    //[155].最小栈
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

    //[剑指 Offer 31].栈的压入、弹出序列
    //[946].验证栈序列
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Stack<Integer> stack = new Stack<>();
        //= [1,2,3,4,5], popped = [4,3,5,1,2]
        int index = 0;
        for (int push : pushed) {
            stack.push(push);
            while (!stack.isEmpty() && stack.peek() == popped[index]) {
                stack.pop();
                index++;
            }
        }
        return stack.isEmpty();
    }

    //[剑指 Offer 33].二叉搜索树的后序遍历序列
    public boolean verifyPostorder(int[] postorder) {
//        return dfsForVerifyPostorder(postorder, 0, postorder.length - 1);

        Stack<Integer> stack = new Stack<>();
        int parent = Integer.MAX_VALUE;
        //4 6 7 5 2 3 1
        //右子树 6 7 5， 左子树 2 3 1
        //i < i+1，意味着 i+1是i的右子树
        //i > i+1, 意味着 i+1是0～i的某一个的左子树
        for (int i = postorder.length - 1; i >= 0; i--) {
            int cur = postorder[i];
            //遇到右子树节点直接进入栈
            //遇到左子树节点则出栈，找到栈中最小的值为父亲节点
            while (!stack.isEmpty() && cur < stack.peek()) {
                parent = stack.pop();
            }
            //因为左子树节点才出栈找到该节点的父亲节点，那么父亲节点 > 左子树节点
            if (parent < cur) {
                return false;
            }
            stack.push(cur);
        }
        return true;
    }

    private boolean dfsForVerifyPostorder(int[] postorder, int i, int j) {
        if (i >= j) return true;
        //找到左子树和右子树区间，如果左子树的所有节点都小于根节点，并且右子树的所有节点都大于根节点
        int root = postorder[j];
        //从左遍历到第一个大的值，即为m节点，然后再次遍历到j-1，理论上应该要和j重合
        int p = i;
        while (postorder[p] < postorder[j]) p++;
        //找到第一个大于的值
        int m = p;
        while (postorder[p] > postorder[j]) p++;
        //正常应该能重合
        if (p != j) {
            return false;
        }
        //左子树为后序遍历且右子树也为后序遍历
        return dfsForVerifyPostorder(postorder, i, m - 1) && dfsForVerifyPostorder(postorder, m, j - 1);
    }

    //[剑指 Offer 36].二叉搜索树与双向链表
    //[426].将二叉搜索树转化为排序的双向链表
    private Node preOne = null, headOne = null;

    public Node treeToDoublyList(Node root) {
        if (root == null) return null;
        dfsForTreeToDoublyList(root);
        preOne.right = headOne;
        headOne.left = preOne;
        return headOne;
    }

    private void dfsForTreeToDoublyList(Node root) {
        if (root == null) return;
        dfsForTreeToDoublyList(root.left);
        if (preOne == null) {
            headOne = root;
        } else {
            preOne.right = root;
        }
        root.left = preOne;
        preOne = root;
        dfsForTreeToDoublyList(root.right);
    }

    //[剑指 Offer 45].把数组排成最小的数
    public String minNumber(int[] nums) {
//        int n = nums.length;
//        String[] str = new String[n];
//        for (int i = 0; i < n; i++) {
//            str[i] = String.valueOf(nums[i]);
//        }
//        Arrays.sort(str, (x, y) -> (x + y).compareTo(y + x));
//
//        StringBuilder sb = new StringBuilder();
//
//        for (String string : str) {
//            sb.append(string);
//        }
//        return sb.toString();

        //面试肯定要手撕快排
        int n = nums.length;
        String[] arr = new String[n];
        for (int i = 0; i < n; i++) {
            arr[i] = String.valueOf(nums[i]);
        }
        quickSort(arr, 0, n - 1);
        StringBuilder sb = new StringBuilder();
        for (String str : arr) {
            sb.append(str);
        }
        return sb.toString();
    }

    private void quickSort(String[] arr, int left, int right) {
        if (left >= right) return;
        int i = left, j = right;

        String pivot = arr[left];
        while (i < j) {
            //一定是跟arr[left]比较
            while (i < j && (arr[j] + arr[left]).compareTo(arr[left] + arr[j]) >= 0) j--;
            arr[i] = arr[j];
            while (i < j && (arr[i] + arr[left]).compareTo(arr[left] + arr[i]) <= 0) i++;
            arr[j] = arr[i];
        }
        arr[i] = pivot;

        quickSort(arr, left, i - 1);
        quickSort(arr, i + 1, right);
    }

    //[剑指 Offer 46].把数字翻译成字符串
    public int translateNum(int num) {
        String s = String.valueOf(num);
        int n = s.length();
        //前面长度为i的字符串有几种拼法
        int[] dp = new int[n + 1];
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            dp[i] = dp[i - 1];
            if (i > 1) {
                //与前面一位组成的两位数字在10和25之间，那么还有从前面获得
                int b = (s.charAt(i - 2) - '0') * 10 + (s.charAt(i - 1) - '0');
                if (b >= 10 && b <= 25) {
                    dp[i] += dp[i - 2];
                }
            }
        }
        return dp[n];
    }

    //[剑指 Offer 51].数组中的逆序对
    public int reversePairs(int[] nums) {
        return mergeSort(nums, 0, nums.length - 1);
    }

    private int merge(int[] nums, int left, int mid, int right) {
        int[] temp = new int[right - left + 1];
        int i = left, j = mid + 1, index = 0;
        int count = 0;
        while (i <= mid && j <= right) {
            if (nums[i] <= nums[j]) {
                temp[index++] = nums[i++];
            } else {
                temp[index++] = nums[j++];
                //后面的
                count += mid - i + 1;
            }
        }

        while (i <= mid) {
            temp[index++] = nums[i++];
        }

        while (j <= right) {
            temp[index++] = nums[j++];
        }

        for (int k = 0; k < right - left + 1; k++) {
            nums[k + left] = temp[k];
        }
        return count;
    }

    private int mergeSort(int[] nums, int left, int right) {
        if (left >= right) return 0;

        int mid = left + (right - left) / 2;

        int a = mergeSort(nums, left, mid);
        int b = mergeSort(nums, mid + 1, right);

        int c = merge(nums, left, mid, right);
        return a + b + c;
    }

    //[剑指 Offer II 053].二叉搜索树中的中序后继
    TreeNode next = null;

    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        traverse(root, p);
        return next;
    }

    private void traverse(TreeNode root, TreeNode p) {
        if (root == null) return;
        traverse(root.left, p);
        if (root.val > p.val && next == null) {
            next = root;
            return;
        }
        traverse(root.right, p);
    }

    //[面试题 03.05].栈排序
    public class SortedStack {

        //stack排序，正常返回
        Stack<Integer> stack, temp;

        public SortedStack() {
            stack = new Stack<>();
            temp = new Stack<>();
        }

        public void push(int val) {
            if (stack.isEmpty()) {
                stack.push(val);
            } else {
                while (!stack.isEmpty() && val > stack.peek()) {
                    temp.push(stack.pop());
                }

                stack.push(val);

                while (!temp.isEmpty()) {
                    stack.push(temp.pop());
                }
            }
        }

        public void pop() {
            if (!stack.isEmpty()) {
                stack.pop();
            }
        }

        public int peek() {
            if (stack.isEmpty()) return -1;
            return stack.peek();
        }

        public boolean isEmpty() {
            return stack.isEmpty();
        }
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

    //[面试题 08.12].八皇后
    public List<List<String>> solveNQueensV2(int n) {
        char[][] board = new char[n][n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(board[i], '.');
        }
        List<List<String>> res = new ArrayList<>();
        dfsForSolveNQueens(n, 0, board, res);
        return res;
    }

    private void dfsForSolveNQueens(int n, int row, char[][] board, List<List<String>> select) {
        if (row == n) {
            List<String> list = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                list.add(String.valueOf(board[i]));
            }
            select.add(list);
            return;
        }

        for (int i = 0; i < n; i++) {
            if (!isValidForSolveNQueens(row, i, board, n)) {
                continue;
            }
            board[row][i] = 'Q';

            dfsForSolveNQueens(n, row + 1, board, select);

            board[row][i] = '.';
        }
    }

    private boolean isValidForSolveNQueens(int x, int y, char[][] board, int n) {
        //因为是行选择，必定是一个合法
        for (int i = 0; i < x; i++) {
            if (board[i][y] == 'Q') {
                return false;
            }
        }
        for (int i = x - 1, j = y + 1; i >= 0 && j < n; i--, j++) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = x - 1, j = y - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }

    //[面试题 10.03].搜索旋转数组
    public int search4(int[] arr, int target) {
        if (arr.length == 0) return -1;
        int left = 0, right = arr.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] > arr[left]) {
                if (arr[left] <= target && target <= arr[mid]) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            } else if (arr[mid] < arr[left]) {
                if (arr[left] <= target || target <= arr[mid]) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            } else {
                if (arr[left] != target) {
                    left++;
                } else {
                    right = left;
                }
            }
        }
        return arr[left] == target ? left : -1;
    }

    //[面试题 10.10].数字流的秩
    class StreamRank {
        //树状数组解决的问题是，频繁变更，求解区间和问题
        //频繁求解<=x的数量，可以通过每个
        //tree是从1开始计数，x范围是5000，所以数组必须是50001个数字
        int[] tree;
        int n;

        public StreamRank() {
            n = 50001;
            //最后一个是空格，防止query的时候，边界为50001，实际数组超了
            tree = new int[n + 1];
        }

        private int lowbit(int x) {
            return x & (-x);
        }

        private int query(int x) {
            int ans = 0;
            for (int i = x; i > 0; i -= lowbit(i)) ans += tree[i];
            return ans;
        }

        private void add(int x, int u) {
            for (int i = x; i <= n; i += lowbit(i)) tree[i] += u;
        }

        public void track(int x) {
            add(x + 1, 1);
        }

        public int getRankOfNumber(int x) {
            return query(x + 1);
        }
    }

    //[面试题 16.19].水域大小
    public int[] pondSizes(int[][] land) {
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < land.length; i++) {
            for (int j = 0; j < land[0].length; j++) {
                if (land[i][j] == 0) {
                    AtomicInteger count = new AtomicInteger();
                    dfsForPondSizes(land, i, j, count);
                    res.add(count.get());
                }
            }
        }
        return res.stream().mapToInt(a -> a).sorted().toArray();
    }

    private void dfsForPondSizes(int[][] land, int x, int y, AtomicInteger count) {
        if (x < 0 || y < 0 || x >= land.length || y >= land[0].length || land[x][y] != 0) {
            return;
        }
        land[x][y] = 2;
        count.incrementAndGet();
        int[][] directs = new int[][]{{0, -1}, {0, 1}, {-1, -1}, {-1, 0}, {-1, 1}, {1, -1}, {1, 0}, {1, 1}};
        for (int[] dir : directs) {
            int nx = x + dir[0];
            int ny = y + dir[1];
            dfsForPondSizes(land, nx, ny, count);
        }
    }

    //[面试题 17.24].最大子矩阵
    public int[] getMaxMatrix(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        //投影，列的和
        int[] sum = new int[n];
        int[] ans = new int[4];
        int maxAns = matrix[0][0];
        //上边界
        for (int i = 0; i < m; i++) {
            //固定了第一行之后，就需要重置
            for (int t = 0; t < n; t++) sum[t] = 0;
            //下边界
            for (int j = i; j < m; j++) {
                // start 记录最大连续数组子序的开始位置
                int maxSum = 0, start = 0;
                //从左到右遍历
                for (int k = 0; k < n; k++) {
                    sum[k] += matrix[j][k];
                    //动态规划求sum数组的最大连续子序列之和
                    if (maxSum > 0) {
                        maxSum += sum[k];
                    } else {
                        maxSum = sum[k];
                        start = k;
                    }

                    if (maxSum > maxAns) {
                        ans[0] = i;
                        ans[1] = start;
                        ans[2] = j;
                        ans[3] = k;
                        maxAns = maxSum;
                    }
                }
            }
        }
        return ans;
    }

    //[补充题13].中文数字转阿拉伯数字
    public static String chinese2Arabic(String zh) {
        HashMap<Character, Long> w2n = new HashMap<Character, Long>() {{
            put('一', 1L);
            put('二', 2L);
            put('三', 3L);
            put('四', 4L);
            put('五', 5L);
            put('六', 6L);
            put('七', 7L);
            put('八', 8L);
            put('九', 9L);
        }};
        HashMap<Character, Long> w2e = new HashMap<Character, Long>() {{
            put('十', 10L);
            put('百', 100L);
            put('千', 1000L);
            put('万', 10000L);
            put('亿', 100000000L);
        }};

        Stack<Long> stack = new Stack<>();
        if (helper(stack, zh, w2n, w2e)) {
            StringBuilder sb = new StringBuilder();
            long ans = 0;
            for (long num : stack) {
                ans += num;
            }
            sb.append(ans);
            return sb.toString();
        }
        return null;
    }

    private static boolean helper(Stack<Long> st, String zh, Map<Character, Long> w2n, Map<Character, Long> w2e) {
        if (zh.length() == 0) {
            return true;
        }
        char ch = zh.charAt(0);
        if (w2e.containsKey(ch)) {
            if (st.isEmpty() || st.peek() >= w2e.get(ch)) {
                return false;
            }
            int tmp = 0;
            while (!st.isEmpty() && st.peek() < w2e.get(ch)) {
                tmp += st.pop();
            }
            st.push(tmp * w2e.get(ch));
            return helper(st, zh.substring(1), w2n, w2e);
        } else if (w2n.containsKey(ch)) {
            st.push(w2n.get(ch));
            return helper(st, zh.substring(1), w2n, w2e);
        } else if (ch == '零') {
            return helper(st, zh.substring(1), w2n, w2e);
        } else {
            return false;
        }
    }

    //Morris前序遍历
    private void preOrderMorris(TreeNode root) {
        TreeNode cur = root, rightMost = null;
        List<Integer> list = new ArrayList<>();
        while (cur != null) {
            if (cur.left == null) {
                list.add(cur.val);
                //是从这边直接遍历到原来的根节点的
                cur = cur.right;
            } else {
                rightMost = cur.left;
                while (rightMost.right != null && rightMost.right != cur) {
                    rightMost = rightMost.right;
                }
                //第一次访问，创建到根节点的快捷路径
                if (rightMost.right == null) {
                    list.add(cur.val);
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

    //Morris中序遍历
    private void inorderMorris(TreeNode root) {
        TreeNode cur = root, rightMost = null;
        List<Integer> res = new ArrayList<>();
        while (cur != null) {
            if (cur.left == null) {
                res.add(cur.val);
                //是从这边直接遍历到原来的根节点的
                cur = cur.right;
            } else {
                rightMost = cur.left;
                while (rightMost.right != null && rightMost.right != cur) {
                    rightMost = rightMost.right;
                }
                //第一次访问，创建到根的快捷路径，意味着第一次访问根节点
                if (rightMost.right == null) {
                    rightMost.right = cur;
                    cur = cur.left;
                } else {
                    //第二次访问根节点，直接断开，说明中序遍历位置
                    res.add(cur.val);
                    rightMost.right = null;
                    cur = cur.right;
                }
            }
        }
    }

    public static void main(String[] args) {
//        System.out.println(maxSlidingWindow(new int[]{1, 3, -1, -3, 5, 3, 6, 7}, 3));
//        System.out.println(mostPoints(new int[][]{{3, 2}, {4, 3}, {4, 4}, {2, 5}}));
//        System.out.println(mostPoints(new int[][]{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}}));
//        System.out.println(Arrays.toString(divideString("abcdefghij", 3, 'x')));
//        System.out.println(minMoves(5, 0));
//        System.out.println(minMoves(19, 2));
//        System.out.println(minMoves(10, 4));
//        System.out.println(new AllOfThem().wordCount(new String[]{"ant", "act", "tack"}, new String[]{"tack", "act", "acti"}));
//        System.out.println(Arrays.toString(new AllOfThem().executeInstructions(3, new int[]{0, 1}, "RRDDLU")));
//        System.out.println(new AllOfThem().permute(new int[]{1, 2, 3}));
//        System.out.println(new AllOfThem().permuteUnique(new int[]{1, 1, 2}));
//
//        System.out.println(new AllOfThem().findMinHeightTrees(2, new int[][]{{0, 1}}));
//        System.out.println(new AllOfThem().findMinHeightTrees(6, new int[][]{{3, 0}, {3, 1}, {3, 2}, {3, 4}, {5, 4}}));
//        System.out.println(new AllOfThem().numTrees(3));
//        System.out.println(new AllOfThem().generate(5));
//        System.out.println(new AllOfThem().getRow(4));
//        System.out.println(new AllOfThem().partition("aabc"));
//        System.out.println(new AllOfThem().convert("PAYPALISHIRING", 4));
//        System.out.println(new AllOfThem().convert("PAYPALISHIRING", 3));
//        System.out.println(new AllOfThem().convert("A", 1));
//
//        System.out.println(new AllOfThem().reverse(120));
//        System.out.println(new AllOfThem().reverse(-123));
//        System.out.println(new AllOfThem().singleNumber(new int[]{1, 2, 1}));
//        System.out.println(new AllOfThem().singleNumber(new int[]{4, 1, 2, 1, 2}));
//        System.out.println(new AllOfThem().singleNumber2(new int[]{0, 1, 0, 1, 0, 1, -99}));
//        ListNode list = new ListNode(1);
//        list.next = new ListNode(2);
//        list.next.next = new ListNode(3);
//        list.next.next.next = new ListNode(4);
//        ListNode r1 = new AllOfThem().swapPairs(list);
//
//        ListNode second = new ListNode(1);
//        second.next = new ListNode(2);
//        second.next.next = new ListNode(3);
//        ListNode result = new AllOfThem().mergeTwoList(list, second);
//
//        ListNode l61 = new ListNode(1);
//        l61.next = new ListNode(2);
//        l61.next.next = new ListNode(3);
//        l61.next.next.next = new ListNode(4);
//        l61.next.next.next.next = new ListNode(5);
//        ListNode r61 = new AllOfThem().rotateRight(l61, 2);
//
//        ListNode l141 = new ListNode(1);
//        l141.next = new ListNode(2);
//        l141.next.next = new ListNode(3);
//        l141.next.next.next = new ListNode(4);
//        l141.next.next.next.next = new ListNode(5);
//        new AllOfThem().reorderList(l141);
//
//        ListNode l328 = new ListNode(1);
//        l328.next = new ListNode(2);
//        l328.next.next = new ListNode(3);
//        l328.next.next.next = new ListNode(4);
//        ListNode r328 = new AllOfThem().oddEvenList(l328);
//
//        System.out.println(new AllOfThem().reverseWords("  Bob    Loves  Alice   "));
//
//        System.out.println(new AllOfThem().findLeftIndex(new int[]{1}, 5));
//        System.out.println(new AllOfThem().searchInsert(new int[]{1}, 5));
//        System.out.println(new AllOfThem().search3(new int[]{1, 0, 1, 1, 1}, 0));
//
//        System.out.println(new AllOfThem().letterCombinations("23"));
//        System.out.println(new AllOfThem().combinationSum(new int[]{2, 3, 4, 7}, 7));
//        System.out.println(new AllOfThem().combinationSum2(new int[]{10, 1, 2, 7, 6, 1, 5}, 8));
//        System.out.println(new AllOfThem().combinationSum3(3, 9));
//        System.out.println(new AllOfThem().subsets(new int[]{1, 2, 3, 4}));
//        System.out.println(new AllOfThem().subsetsWithDup(new int[]{1, 2, 2}));
//        System.out.println(new AllOfThem().trap(new int[]{0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1}));
//        System.out.println(new AllOfThem().trap(new int[]{4, 2, 0, 3, 2, 5}));
//        System.out.println(Arrays.toString(new AllOfThem().nextGreaterElement(new int[]{4, 1, 2}, new int[]{1, 3, 4, 2})));
//        System.out.println(new AllOfThem().combine(4, 2));
//        System.out.println(new AllOfThem().solveNQueens(4));
//        System.out.println(new AllOfThem().find132pattern(new int[]{1,2,3,4}));
//        System.out.println(new AllOfThem().find132pattern(new int[]{3,1,4,2}));
//        System.out.println(new AllOfThem().find132pattern(new int[]{-1, 3, 2, 0}));
//        System.out.println(new AllOfThem().find132pattern(new int[]{3, 5, 0, 3, 4}));
//        System.out.println(Arrays.toString(new AllOfThem().dailyTemperatures(new int[]{30, 60, 90})));
//        System.out.println(Arrays.toString(new AllOfThem().dailyTemperatures(new int[]{73, 74, 75, 71, 69, 72, 76, 73})));
//        System.out.println(Arrays.toString(new AllOfThem().dailyTemperatures(new int[]{73})));
//        System.out.println(new AllOfThem().findUnsortedSubarray(new int[]{2, 6, 4, 8, 10, 9, 15}));
//        System.out.println(new AllOfThem().findUnsortedSubarray(new int[]{1, 2, 3, 4}));
//        System.out.println(new AllOfThem().findUnsortedSubarray(new int[]{1, 2, 3, 3, 3}));
//        System.out.println(new AllOfThem().findUnsortedSubarray(new int[]{1}));
//        System.out.println(new AllOfThem().grayCode(3));
//        System.out.println(new AllOfThem().superPow(2, new int[]{1, 0}));
//        System.out.println(new AllOfThem().trailingZeroes(10000));
//
//        TreeNode t199 = new TreeNode(1);
//        t199.left = new TreeNode(2);
//        t199.right = new TreeNode(3);
//        t199.left.right = new TreeNode(5);
//        t199.right.left = new TreeNode(4);
//        System.out.println(new AllOfThem().rightSideView(t199));
//        System.out.println(new AllOfThem().merge(new int[][]{{1, 4}, {1, 4}}));
//
//        System.out.println(new AllOfThem().numSquares(12));
//
//        TopVotedCandidate candidate = new AllOfThem.TopVotedCandidate(new int[]{0, 1, 1, 0, 0, 1, 0}, new int[]{0, 5, 10, 15, 20, 25, 30});
//        System.out.println(candidate.q(3));
//        System.out.println(candidate.q(12));
//        System.out.println(candidate.q(25));
//        System.out.println(candidate.q(15));
//        System.out.println(candidate.q(24));
//        System.out.println(candidate.q(8));
//        System.out.println(new AllOfThem().jump(new int[]{2, 3, 1, 1, 4}));
//
//        System.out.println(new AllOfThem().simplifyPath("../"));
//        System.out.println(new AllOfThem().timeRequiredToBuy(new int[]{2, 3, 4, 1}, 1));
//        System.out.println(new AllOfThem().minEatingSpeed(new int[]{30, 11, 23, 4, 20}, 6));
//        System.out.println(new AllOfThem().shipWithinDays(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 5));
//        System.out.println(new AllOfThem().mySqrt(2147395599));
//
//        TreeNode t1712 = new TreeNode(4);
//        t1712.left = new TreeNode(2);
//        t1712.right = new TreeNode(5);
//        t1712.left.left = new TreeNode(1);
//        t1712.left.right = new TreeNode(3);
//        t1712.right.right = new TreeNode(6);
//        t1712.left.left.left = new TreeNode(0);
//
//        new AllOfThem().convertBiNode(t1712);
//
//        System.out.println(new AllOfThem().spiralOrder(new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}));
//        System.out.println(new AllOfThem().generateMatrix(3));
//
//        NumArray numArray = new NumArray(new int[]{-2, 0, 3, -5, 2, -1});
//        System.out.println(numArray.sumRange(0, 2));
//        System.out.println(numArray.sumRange(2, 5));
//        System.out.println(numArray.sumRange(0, 5));
//
//        System.out.println(Arrays.toString(new AllOfThem().corpFlightBookings(new int[][]{{1, 2, 10}, {2, 3, 20}, {2, 5, 25}}, 5)));
//        System.out.println(new AllOfThem().countBattleships(new char[][]{{'X', '.', '.'}, {'X', 'X', 'X'}, {'X', '.', '.'}}));
//        System.out.println(new AllOfThem().countBattleshipsV2(new char[][]{{'X', '.', '.'}, {'X', 'X', 'X'}, {'X', '.', '.'}}));
//        System.out.println(new AllOfThem().findRepeatedDnaSequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"));
//        System.out.println(new AllOfThem().decodeCiphertext("ch   ie   pr", 3));
//        System.out.println(new AllOfThem().decodeCiphertext("coding", 1));
//        System.out.println(new AllOfThem().getDescentPeriods(new int[]{12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 4, 3, 10, 9, 8, 7}));
//        System.out.println(new AllOfThem().kIncreasing(new int[]{5, 5, 5, 5, 4}, 1));
//        System.out.println(new AllOfThem().findLeftIndex2(new int[]{1, 2, 2, 4}, 2));
//        System.out.println(new AllOfThem().findLeftIndex2(new int[]{1, 2, 2, 4}, 3));
//        System.out.println(new AllOfThem().findRightIndexV2(new int[]{1, 2, 2, 4}, 2));
//        System.out.println(new AllOfThem().findRightIndexV2(new int[]{1, 2, 2, 4}, 3));
//        System.out.println(new AllOfThem().findRightIndexV3(new int[]{1, 2, 2, 4}, 2));
//        System.out.println(new AllOfThem().findRightIndexV3(new int[]{1, 2, 2, 4}, 3));
//        System.out.println(new AllOfThem().isAdditiveNumber("199100199"));
//        System.out.println(new AllOfThem().nthUglyNumber(10));
//        System.out.println(new AllOfThem().eventualSafeNodes(new int[][]{{1, 2}, {2, 3}, {5}, {0}, {5}, {}, {}}));
//        System.out.println(new AllOfThem().eventualSafeNodes(new int[][]{{1, 2, 3, 4}, {1, 2}, {3, 4}, {0, 4}, {}}));
//        System.out.println(new AllOfThem().invalidTransactions(new String[]{"alice,20,1220,mtv", "alice,20,1220,mtv"}));
//        System.out.println(new AllOfThem().countGoodSubstrings("xyzzaz"));
//        System.out.println(new AllOfThem().countGoodSubstrings("aababcabc"));
//        System.out.println(new AllOfThem().intToRoman(3));
//        System.out.println(new AllOfThem().longestCommonPrefix(new String[]{"flower", "flow", "flight"}));
//        System.out.println(new AllOfThem().longestCommonPrefix(new String[]{"dog", "racecar", "car"}));
//        System.out.println(new AllOfThem().countQuadruplets(new int[]{1, 2, 3, 6}));
//        System.out.println(new AllOfThem().countQuadruplets(new int[]{1, 1, 1, 3, 5}));
//        System.out.println(new AllOfThem().removeElement(new int[]{1}, 1));
//        System.out.println(new AllOfThem().removeElement(new int[]{1}, 0));
//        System.out.println(new AllOfThem().addBinary("11", "1"));
//        System.out.println(new AllOfThem().restoreIpAddresses("25525511135"));
//        System.out.println(new AllOfThem().restoreIpAddresses("101023"));
//        System.out.println(new AllOfThem().restoreIpAddresses("010010"));
//        System.out.println(new AllOfThem().findCircleNum(new int[][]{{1, 1, 0}, {1, 1, 0}, {0, 0, 1}}));
//        System.out.println(new AllOfThem().grayCode(3));
//        System.out.println(new AllOfThem().minSwaps(new int[]{0, 1, 0, 1, 1, 0, 0}));
//        System.out.println(new AllOfThem().minSwaps(new int[]{0, 1, 1, 1, 0, 0, 1, 1, 0}));
//        System.out.println(new AllOfThem().minSwaps(new int[]{1, 1, 0, 0, 1}));
//        System.out.println(new AllOfThem().isInterleave("aabcc", "dbbca", "aadbbcbcac"));
//        System.out.println(new AllOfThem().isInterleave("aabcc", "dbbca", "aadbbbaccc"));
//        System.out.println(new AllOfThem().isInterleave("", "", ""));
//        TreeNode t623 = new TreeNode(4);
//        t623.left = new TreeNode(2);
//        t623.left.left = new TreeNode(3);
//        t623.left.right = new TreeNode(1);
//        t623.right = new TreeNode(6);
//        t623.right.left = new TreeNode(5);
//        System.out.println(new AllOfThem().addOneRow(t623, 1, 2));
//        compress(new char[]{'a', 'a', 'b', 'b', 'c', 'c', 'c'});
//        System.out.println(new AllOfThem().diffWaysToCompute("2*3-4*5"));
//        System.out.println(findNthDigit(3));
//        System.out.println(lengthLongestPath("dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"));
//        System.out.println(lengthLongestPath("file1.txt\nfile2.txt\nlongfile.txt"));
//        System.out.println(readBinaryWatch(1));
//        System.out.println(addSpaces("spacing", new int[]{0,1,2,3,4,5,6}));
//        System.out.println(longestPalindromeSubseq("bbbab"));
//        System.out.println(magicalString(6));
//        System.out.println(new AllOfThem().numberOfSteps(8));
//        System.out.println(PredictTheWinner(new int[]{1, 5, 233, 7}));
//        System.out.println(minWindow("ADOBECODEBANC", "ABC"));
//        System.out.println(ladderLength("hit", "cog", Arrays.asList("hot", "dot", "dog", "lot", "log", "cog")));
//        System.out.println(convertToTitle(701));
//        System.out.println(canPartitionKSubsets(new int[]{4, 3, 2, 3, 5, 2, 1}, 4));
//        System.out.println(subarraysDivByK(new int[] {-1, 2 ,1}, 2));

//        System.out.println(isMatch("aab", "c*a*b"));
//        System.out.println(isMatch("mississippi", "mis*is*p*."));
//        System.out.println(longestDiverseString(1, 1, 7));
//        System.out.println(calculate("(1+(4+5+2)-3)+(6+8)"));
//        System.out.println(calculate("3 * (4-5 /2) -6"));
//        System.out.println(new AllOfThem().networkDelayTime(new int[][]{{2, 1, 1}, {2, 3, 1}, {3, 4, 1}}, 4, 2));

//        System.out.println(backspaceCompare("y#fo##f", "y#f#o##f"));
//        System.out.println(largestMultipleOfThree(new int[]{0, 0, 0, 0, 0, 0}));
//        System.out.println(largestMultipleOfThree(new int[]{8, 1, 9}));
//        System.out.println(largestMultipleOfThree(new int[]{8, 6, 7, 1, 0}));
//
//        System.out.println(missingElement(new int[]{4, 7, 9, 10}, 1));
//        System.out.println(missingElement(new int[]{1, 2, 4}, 3));
//        System.out.println(missingElement(new int[]{4, 7, 9, 10}, 3));
//        System.out.println(singleNonDuplicate(new int[]{1, 1, 2, 3, 3}));
//        ListNode l1 = new ListNode(1);
//        l1.next = new ListNode(2);
//        l1.next.next = new ListNode(3);
//        l1.next.next.next = new ListNode(4);
//        l1.next.next.next.next = new ListNode(5);
//        new AllOfThem().reverseKGroup(l1, 2);

        System.out.println(new AllOfThem().nearestPalindromic("123"));
        System.out.println(new AllOfThem().nearestPalindromic("99"));
        System.out.println(new AllOfThem().nearestPalindromic("100"));
    }
}
