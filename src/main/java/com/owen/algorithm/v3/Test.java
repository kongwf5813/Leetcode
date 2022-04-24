package com.owen.algorithm.v3;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Stack;

import com.owen.algorithm.v3.AllOfThem.*;



public class Test {


    public static void main(String[] args) {
//        System.out.println(numSquares(7));
//        System.out.println(missingElement(new int[] {4,7,9, 10}, 1));
//        System.out.println(missingElement(new int[] {4,7,9, 10}, 3));
//        System.out.println(missingElement(new int[] {1,2,4}, 3));

//        List<List<Interval>> res = new ArrayList<>();
//        res.add(Arrays.asList(new Interval(1,2), new Interval(5,6)));
//        res.add(Arrays.asList(new Interval(1,3)));
//        res.add(Arrays.asList(new Interval(4,10)));
//        System.out.println(employeeFreeTime(res));

        longestPalindrome("abba");
    }

    public static int numSquares(int n) {
        if (n <= 0) return 0;
        int[] dp = new int[n + 1];
        Arrays.fill(dp, 0x3f3f3f3f);
        dp[0] = 0;
        //物品是完全平方数
        for (int i = 1; i * i <= n; i++) {
            //背包是n
            for (int j = i * i; j <= n; j++) {
                dp[j] = Math.min(dp[j - i * i] + 1, dp[j]);
            }
        }
        return dp[n];
    }

    public static void morrisPreorder(TreeNode root) {
        TreeNode cur = root, rightMost = null;

        List<Integer> res = new ArrayList<>();
        while (cur != null) {
            if (cur.left == null) {
                res.add(cur.val);
                cur = cur.right;
            } else {
                rightMost = cur.left;
                while (rightMost.right != null && rightMost.right != cur) {
                    rightMost = rightMost.right;
                }
                if (rightMost == null) {
                    res.add(cur.val);
                    rightMost.right = cur;
                    cur = cur.left;
                } else {
                    rightMost.right = null;
                    cur = cur.right;
                }
            }
        }
    }


    class NumArray307 {

        int[] tree;
        int[] nums;

        public int lowbit(int x) {
            return x & (-x);
        }

        public void add(int index, int v) {
            for (int i = index; i < tree.length; i += lowbit(i)) {
                tree[i] += v;
            }
        }

        public int query(int index) {
            int ans = 0;
            for (int i = index; i > 0; i -= lowbit(i)) {
                ans += tree[i];
            }
            return ans;
        }

        public NumArray307(int[] nums) {
            int n = nums.length;
            tree = new int[n + 1];
            this.nums = nums;
        }


    }

    class Difference {

        int[] diff;

        public Difference(int[] nums) {
            int n = nums.length;
            diff = new int[n];
            int temp = 0;
            for (int i = 0; i < n; i++) {
                diff[i] = nums[i] - temp;
                temp = nums[i];
            }
        }

        public void insert(int i, int j, int value) {
            diff[i] += value;
            if (j + 1 < diff.length) {
                diff[j + 1] -= value;
            }
        }

        public int[] result() {
            int[] res = new int[diff.length];
            int sum = 0;
            for (int i = 0; i < diff.length; i++) {
                sum += diff[i];
                res[i] = sum;
            }
            return res;
        }
    }


    public static int missingElement(int[] nums, int k) {
        int n = nums.length;
        //边界情况
        if (missingCount(nums, n - 1) < k) {
            return nums[n - 1] + k - missingCount(nums, n - 1);
        }

        int left = 0, right = n - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (missingCount(nums, mid) < k) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return nums[left - 1] + k - missingCount(nums, left - 1);
    }

    private static int missingCount(int[] nums, int i) {
        return (nums[i] - nums[0]) - (i - 0);
    }

    public static String serialize(Node node) {
        StringBuilder sb = new StringBuilder();
        dfsForSerialize(node, sb);
        return sb.toString();
    }

    private static void dfsForSerialize(Node node, StringBuilder sb) {
        if (node == null) return;

        sb.append(node.val).append('_');
        sb.append(node.children.size()).append('_');

        for (Node child : node.children) {
            dfsForSerialize(child, sb);
        }
    }

    public static Node deserialize(String data) {
        if (data == null || data.length() == 0) return null;
        String[] split = data.split("_");
        LinkedList<String> list = new LinkedList<>();
        for (String str : split) {
            list.addLast(str);
        }
        return dfsForDeserialize(list);
    }

    private static Node dfsForDeserialize(LinkedList<String> list) {
        if (list == null || list.isEmpty()) return null;

        int val = Integer.parseInt(list.removeFirst());
        int size = Integer.parseInt(list.removeFirst());
        Node node = new Node(val);
        node.children = new ArrayList<>();
        for (int i = 0; i < size ; i++) {
            node.children.add(dfsForDeserialize(list));
        }
        return node;
    }

    static class Interval {
        int start;
        int end;
        public Interval(int start, int end) {
            this.start = start;
            this.end = end;
        }
    }

    public static List<Interval> employeeFreeTime(List<List<Interval>> schedule) {
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
            if (cur.start > end) {
                res.add(new Interval(end, cur.start));
            }
            end = Math.max(cur.end, end);
        }
        return res;
    }

    public static String longestPalindrome(String s) {
        int n = s.length();
        int maxLen = 0, begin = 0;
        for (int i = 0; i < n; i++) {
            //奇数
            int len1 = expand(s, i, i);
            //偶数
            int len2 = expand(s, i, i +1);
            int len = Math.max(len1, len2);
            if (len > maxLen) {
                maxLen = len;
                //有偶数的情况，综合取len-1
                begin = i - (len - 1)/2;
            }
        }
        return s.substring(begin, begin + maxLen);
    }

    private static int expand(String s, int left , int right) {
        while(left >=0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        //不想等
        return right - left + 1 - 2;
    }


    public static String chinese2Arabic(String zh) {
        Map<Character, Long> w2n = new HashMap<Character, Long>() {{
            put('一', 1L);
            put('二', 1L);
            put('三', 1L);
            put('四', 1L);
            put('五', 1L);
            put('六', 1L);
            put('七', 1L);
            put('八', 1L);
            put('九', 1L);
        }};

        Map<Character, Long> w2e = new HashMap<Character, Long>() {{
            put('十', 10L);
            put('百', 100L);
            put('千', 1000L);
            put('万', 10000L);
            put('亿', 100000000L);
        }};
        Stack<Long> stack = new Stack<>();
        if (helper(stack, zh, w2n, w2e)) {
            StringBuilder sb = new StringBuilder();
            long temp = 0;
            while (!stack.isEmpty()) {
                temp += stack.pop();
            }
            sb.append(temp);
            return sb.toString();
        }
        return null;
    }

    private static boolean helper(Stack<Long> stack, String zh, Map<Character, Long> w2n, Map<Character, Long> w2e) {
        if (zh.length() == 0) return true;
        char ch = zh.charAt(0);
        if (w2e.containsKey(ch)) {
            if (stack.isEmpty() || stack.peek() >= w2e.get(ch)) return false;
            int temp = 0;
            while (!stack.isEmpty() && stack.peek() < w2e.get(ch)) {
                temp += stack.pop();
            }
            stack.push(temp * w2e.get(ch));
            return helper(stack, zh.substring(1), w2n, w2e);
        } else if (w2n.containsKey(ch)) {
            stack.push(w2n.get(ch));
            return helper(stack, zh.substring(1), w2n, w2e);
        } else if (ch == '零') {
            return helper(stack, zh.substring(1), w2n, w2e);
        } else {
            return false;
        }
    }
}