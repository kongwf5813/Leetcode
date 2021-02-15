package com.owen.algorithm;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import com.amazonaws.services.dynamodbv2.xspec.S;
import com.owen.algorithm.LinkList.ListNode;

public class Others {

    public static class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;
        public Node random;
        public List<Node> neighbors;

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

    //[6].Z字形变化
    public static String convert(String s, int numRows) {
        if (numRows == 0) return "";
        if (numRows == 1) return s;

        StringBuilder sb = new StringBuilder();
        int index = 0, flag = -1;
        String[] results = new String[numRows];
        for (int i = 0; i < numRows; i++) results[i] = "";
        while (index != s.length()) {
            if (index % (numRows - 1) == 0) {
                flag = -flag;
            }
            if (flag == -1) {
                int real = numRows - 1 - (index % (numRows - 1));
                results[real] += s.charAt(index);
            } else {
                int real = index % (numRows - 1);
                results[real] += s.charAt(index);
            }

            index++;
        }
        for (String result : results) {
            sb.append(result);
        }
        return sb.toString();
    }

    //[11].盛最多水的容器
    public static int maxArea(int[] height) {
        //Math.min(height[j], height[i]) * (j-i)
        //移动最长木板导致面积减小或不变
        //移动最短木板导致面积可能会变大
        int left = 0, right = height.length - 1;
        int res = 0;
        while (left < right) {
            int area = Math.min(height[left], height[right]) * (right - left);
            res = Math.max(area, res);
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return res;
    }

    //[28]. 实现strStr()
    public static int strStr(String haystack, String needle) {
        if (needle == null || needle.length() == 0) {
            return 0;
        }
        if (haystack == null || haystack.length() == 0 || needle.length() > haystack.length()) {
            return -1;
        }
        int l = 0;
        int r = 0;
        while (l < haystack.length() && r < needle.length()) {
            if (needle.charAt(r) == haystack.charAt(l)) {
                r++;
                l++;
            } else {
                l = l - r + 1;
                r = 0;
            }
        }
        if (r == needle.length()) {
            return l - needle.length();
        } else {
            return -1;
        }
    }

    //[42].接雨水
    public static int trap(int[] height) {
        if (height.length == 0) return 0;
        int left = 0, right = height.length - 1;
        //l_max指[0...left]的最大左边界值, r_max指[right...n-1]的最大右边界值
        int l_max = height[0], r_max = height[height.length - 1];
        int res = 0;
        while (left <= right) {
            l_max = Math.max(l_max, height[left]);
            r_max = Math.max(r_max, height[right]);
            if (l_max < r_max) {
                res += l_max - height[left];
                left++;
            } else {
                res += r_max - height[right];
                right--;
            }
        }
        return res;
    }

    //[43].字符串相乘
    public static String multiply(String num1, String num2) {
        char[] c1 = num1.toCharArray();
        char[] c2 = num2.toCharArray();
        int outLevel = 1;
        int res = 0;
        for (int i = c2.length - 1; i >= 0; i--) {
            int cur = 0, innerLevel = 1;
            for (int j = c1.length - 1; j >= 0; j--) {
                cur += (c2[i] - '0') * (c1[j] - '0') * innerLevel;
                innerLevel *= 10;
            }
            res += cur * outLevel;
            outLevel *= 10;
        }
        return "" + res;
    }

    //[49] 字母异位分组
    public static List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> groupMap = new HashMap<>();
        for (String str : strs) {
            char[] sorted = str.toCharArray();
            Arrays.sort(sorted);
            String key = String.valueOf(sorted);
            List<String> value = groupMap.get(key);
            if (value == null) {
                value = new ArrayList<>();
                groupMap.put(key, value);
            }
            value.add(str);
        }
        List<List<String>> result = new ArrayList<>();
        for (List<String> val : groupMap.values()) {
            result.add(val);
        }
        return result;
    }

    //[116].填充每个节点的下一个右侧节点指针
    public static Node connect(Node root) {
        if (root == null) return null;
        dfsForConnect(root.left, root.right);
        return root;
    }

    private static void dfsForConnect(Node left, Node right) {
        if (left == null || right == null) return;
        left.next = right;
        dfsForConnect(left.left, left.right);
        dfsForConnect(right.left, right.right);
        dfsForConnect(left.right, right.left);
    }

    //[117].填充每个节点的下一个右侧节点指针II
    public static Node connect2(Node root) {
        if (root == null) return null;
        dfsForConnect2(root, null);
        return root;
    }

    private static void dfsForConnect2(Node left, Node right) {
        if (left != null) {
            left.next = right;
            Node nextRight = null, next = right;
            while (next != null) {
                if (next.left != null) {
                    nextRight = next.left;
                    break;
                }
                if (next.right != null) {
                    nextRight = next.right;
                    break;
                }
                next = next.next;
            }
            if (left.right != null) {
                dfsForConnect2(left.right, nextRight);
                dfsForConnect2(left.left, left.right);
            } else {
                dfsForConnect2(left.left, nextRight);
            }
        }
    }

    //[127].单词接龙
    public static int ladderLength(String beginWord, String endWord, List<String> wordList) {
        if (!wordList.contains(endWord)) return 0;
        Queue<String> q1 = new ArrayDeque<>();
        q1.add(beginWord);
        Set<String> visited = new HashSet<>();
        visited.add(beginWord);
        int res = 1;
        while (!q1.isEmpty()) {
            int size = q1.size();
            for (int i = 0; i < size; i++) {
                String selected = q1.poll();
                for (String word : wordList) {
                    if (visited.contains(word)) {
                        continue;
                    }
                    if (!canConvert(selected, word)) {
                        continue;
                    }
                    if (word.equals(endWord)) {
                        return res + 1;
                    }
                    q1.offer(word);
                    visited.add(word);
                }
            }
            res += 1;
        }
        return 0;
    }

    private static boolean canConvert(String source, String target) {
        if (source.length() != target.length()) return false;
        char[] sChars = source.toCharArray();
        char[] tChars = target.toCharArray();
        int difCount = 0;
        for (int i = 0; i < source.length(); i++) {
            if (sChars[i] != tChars[i]) {
                difCount += 1;
            }
        }
        return difCount == 1;
    }

    //[136]只出现一次的数字
    public static int singleNumber(int[] nums) {
        int res = nums[0];
        for (int i = 1; i < nums.length; i++) {
            res ^= nums[i];
        }
        return res;
    }

    //[137]只出现一次的数字II
    public static int singleNumber2(int[] nums) {
        int res = 0;
        for (int i = 0; i < 32; i++) {
            int bit = 0;
            for (int j = 0; j < nums.length; j++) {
                bit += (nums[j] >>> i) & 1;
            }
            res |= (bit % 3) << i;
        }
        return res;
    }

    //[138]复制带随机指针的链表
    public static Node copyRandomList(Node head) {
        Map<Node, Node> newMap = new HashMap<>();
        Node cur = head;
        while (cur != null) {
            Node newOne = new Node(cur.val);
            newMap.put(cur, newOne);
            cur = cur.next;
        }
        cur = head;
        while (cur != null) {
            Node newOne = newMap.get(cur);
            Node newLink = newMap.get(cur.random);
            newOne.next = newMap.get(cur.next);
            newOne.random = newLink;
            cur = cur.next;
        }
        return newMap.get(head);
    }

    //[146].LRU缓存机制
    public static class LRUCache {
        LinkedHashMap<Integer, Integer> cache = new LinkedHashMap<>();
        int size;

        public LRUCache(int capacity) {
            this.size = capacity;
        }

        public int get(int key) {
            if (!cache.containsKey(key)) {
                return -1;
            }
            Integer value = cache.get(key);
            cache.remove(key);
            cache.put(key, value);
            return value;
        }

        public void put(int key, int value) {
            if (cache.containsKey(key)) {
                cache.remove(key);
                cache.put(key, value);
                return;
            }

            if (cache.size() >= this.size) {
                cache.remove(cache.keySet().iterator().next());
            }
            cache.put(key, value);
        }
    }

    //[130].被围绕的区域
    public static void solve(char[][] board) {
        if (board.length == 0 || board[0].length == 0) return;
        for (int i = 0; i < board.length; i++) {
            dfsForSolve(board, i, 0);
            dfsForSolve(board, i, board[0].length - 1);
        }

        for (int i = 0; i < board[0].length; i++) {
            dfsForSolve(board, 0, i);
            dfsForSolve(board, board.length - 1, i);
        }

        //边缘的0先变成#，再变成0，而内部的0，变成X
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == '#') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }
    }

    private static void dfsForSolve(char[][] board, int i, int j) {
        //一定要加#的判断，否则导致剪支失败
        if (i < 0 || i > board.length - 1
                || j < 0 || j > board[0].length - 1
                || board[i][j] == 'X'
                || board[i][j] == '#') {
            return;
        }
        board[i][j] = '#';
        dfsForSolve(board, i + 1, j);
        dfsForSolve(board, i - 1, j);
        dfsForSolve(board, i, j + 1);
        dfsForSolve(board, i, j - 1);
    }

    //[131]克隆图
    public static Node cloneGraph(Node node) {
        if (node == null) return null;
        Node head = new Node(node.val);
        Map<Node, Node> map = new HashMap<>();
        Map<Node, Boolean> visited = new HashMap<>();
        map.put(node, head);

        Queue<Node> que = new ArrayDeque<>();
        que.add(node);
        while (!que.isEmpty()) {
            int size = que.size();
            for (int i = 0; i < size; i++) {
                Node top = que.poll();
                if (Boolean.TRUE.equals(visited.get(top))) {
                    continue;
                }

                visited.put(top, Boolean.TRUE);
                map.putIfAbsent(top, new Node(top.val));

                Node copy = map.get(top);
                if (top.neighbors != null) {
                    List<Node> copyNeighbors = new ArrayList<>();
                    for (Node neigh : top.neighbors) {
                        map.putIfAbsent(neigh, new Node(neigh.val));

                        copyNeighbors.add(map.get(neigh));
                        que.add(neigh);
                    }
                    copy.neighbors = copyNeighbors;
                }
            }
        }
        return head;
    }

    //[151].翻转字符串里的单词
    public static String reverseWords(String s) {
        String words[] = s.trim().split("[\\s\\u00A0]+");
        StringBuilder sb = new StringBuilder();
        for (String word : words) {
            sb.insert(0, word + ' ');
        }
        if (sb.length() > 0) sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
    }

    //[165].比较版本号
    public static int compareVersion(String version1, String version2) {
        String[] ves1 = version1.split("\\.");
        String[] ves2 = version2.split("\\.");
        int maxLen = Math.max(ves1.length, ves2.length);
        for (int i = 0; i < maxLen; i++) {
            int v1 = i > ves1.length - 1 ? 0 : Integer.parseInt(ves1[i]);
            int v2 = i > ves2.length - 1 ? 0 : Integer.parseInt(ves2[i]);
            if (v1 > v2) {
                return 1;
            } else if (v1 < v2) {
                return -1;
            }
        }
        return 0;
    }

    //[166].分数到小数
    public static String fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) return "0";
        StringBuilder sb = new StringBuilder();
        if (numerator < 0 ^ denominator < 0) {
            sb.append('-');
        }
        long num = Math.abs(Long.valueOf(numerator));
        long div = Math.abs(Long.valueOf(denominator));
        sb.append(num / div);
        long remain = num % div;
        if (remain == 0)
            return sb.toString();

        sb.append('.');
        HashMap<Long, Integer> remainPos = new HashMap<>();
        while (remain != 0) {
            if (remainPos.containsKey(remain)) {
                int pos = remainPos.get(remain);
                sb.insert(pos, '(');
                sb.append(')');
                break;
            } else {
                remainPos.put(remain, sb.length());
                num = remain * 10;
                sb.append(num / div);
                remain = num % div;
            }
        }
        return sb.toString();
    }

    //[187].重复的DNA序列
    public static List<String> findRepeatedDnaSequences(String s) {
        if (s.length() <= 10) return new ArrayList<>();

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

    //[200].岛屿的数量
    public static int numIslands(char[][] grid) {
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    dfsForNumIslands(grid, i, j);
                    res++;
                }
            }
        }
        return res;
    }

    private static void dfsForNumIslands(char[][] grid, int i, int j) {
        if (i < 0 || i > grid.length - 1 || j < 0 || j > grid[0].length - 1 || grid[i][j] == '0') {
            return;
        }
        grid[i][j] = '0';

        dfsForNumIslands(grid, i + 1, j);
        dfsForNumIslands(grid, i - 1, j);
        dfsForNumIslands(grid, i, j + 1);
        dfsForNumIslands(grid, i, j - 1);
    }

    //[201].数字范围按位与
    public static int rangeBitwiseAnd(int m, int n) {
        int count = 0;
        while (m < n) {
            count++;
            m >>= 1;
            n >>= 1;
        }
        return m << count;
    }

    //[207].课程表
    public static boolean canFinish(int numCourses, int[][] prerequisites) {
        //入度表
        int[] indegrees = new int[numCourses];
        for (int[] pre : prerequisites) {
            indegrees[pre[0]]++;
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < indegrees.length; i++) {
            if (indegrees[i] == 0) {
                queue.offer(i);
            }
        }
        int count = 0;
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            count++;
            for (int[] prereq : prerequisites) {
                //如果节点是 == 另一个节点的入度节点，再进行选择
                if (prereq[1] != cur) continue;

                //将入度减一
                indegrees[prereq[0]]--;
                //如果入度为0
                if (indegrees[prereq[0]] == 0) {
                    queue.offer(prereq[0]);
                }
            }
        }
        return count == numCourses;
    }

    //[210].课程表II
    public static int[] findOrder(int numCourses, int[][] prerequisites) {
        //p[], p[0] <- p[1]
        int[] indegrees = new int[numCourses];
        for (int[] p : prerequisites) {
            indegrees[p[0]]++;
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegrees[i] == 0) {
                queue.offer(i);
            }
        }

        int[] res = new int[numCourses];
        int i = 0;
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            res[i++] = cur;
            for (int[] p : prerequisites) {
                if (p[1] != cur) continue;
                indegrees[p[0]]--;
                if (indegrees[p[0]] == 0) {
                    queue.offer(p[0]);
                }
            }
        }
        return i == numCourses ? res : new int[0];
    }

    //[223].矩形面积
    public static int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        int area1 = (C - A) * (D - B);
        int area2 = (G - E) * (H - F);
        //不相交
        if (E > C || A > G || F > D || B > H) {
            return area1 + area2;
        }

        int dupWidth = Math.min(C, G) - Math.max(A, E);
        int dupHeight = Math.min(D, H) - Math.max(B, F);
        return area1 + area2 - dupHeight * dupWidth;
    }

    //[241].为运算表达式设计优先级
    public static List<Integer> diffWaysToCompute(String input) {
        return dfsForDiffWaysToCompute(input, new HashMap<>());
    }

    private static List<Integer> dfsForDiffWaysToCompute(String input, Map<String, List<Integer>> map) {
        if (map.containsKey(input)) {
            return map.get(input);
        }

        List<Integer> res = new ArrayList<>();
        char[] chars = input.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            char ch = chars[i];
            //开始分治
            if (ch == '+' || ch == '-' || ch == '*') {
                //递归调用左边的运算式子
                List<Integer> left = dfsForDiffWaysToCompute(input.substring(0, i), map);
                //递归调用右边的运算式子
                List<Integer> right = dfsForDiffWaysToCompute(input.substring(i + 1), map);
                for (int l : left) {
                    for (int r : right) {
                        if (ch == '+') {
                            res.add(l + r);
                        } else if (ch == '-') {
                            res.add(l - r);
                        } else {
                            res.add(l * r);
                        }
                    }
                }
            }
        }

        //意味着全是数字
        if (res.size() == 0) {
            res.add(Integer.parseInt(input));
        }

        map.put(input, res);
        return res;
    }

    //[260].只出现一次的数字III
    public static int[] singleNumber3(int[] nums) {
        int xor = 0;
        for (int num : nums) {
            xor ^= num;
        }
        //因为是异或，最低位上的1，肯定表示某一位是不相同的，分成两个组，各自异或一次
        int mask = xor & (-xor);
        int res1 = 0;
        int res2 = 0;
        for (int num : nums) {
            if ((num & mask) == 0) {
                res1 ^= num;
            } else {
                res2 ^= num;
            }
        }
        return new int[]{res1, res2};
    }

    //[263].丑数
    public static boolean isUgly(int num) {
        if (num < 1) return false;
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

    //[264].丑数II
    public static int nthUglyNumber(int n) {
        int p2 = 0, p3 = 0, p5 = 0;
        int[] dp = new int[n];
        dp[0] = 1;
        for (int i = 1; i < n; i++) {
            dp[i] = Math.min(dp[p2] * 2, Math.min(dp[p3] * 3, dp[p5] * 5));
            if (dp[i] == dp[p2] * 2) p2++;
            if (dp[i] == dp[p3] * 3) p3++;
            if (dp[i] == dp[p5] * 5) p5++;
        }
        return dp[n - 1];
    }

    //[295].数据流的中位数
    public static class MedianFinder {

        private PriorityQueue<Integer> small;
        private PriorityQueue<Integer> large;

        /**
         * initialize your data structure here.
         */
        public MedianFinder() {
            //大顶堆
            small = new PriorityQueue<>((a, b) -> b - a);
            //小顶堆
            large = new PriorityQueue<>();
        }

        public void addNum(int num) {
            if (small.size() <= large.size()) {
                large.offer(num);
                small.offer(large.poll());
            } else {
                small.offer(num);
                large.offer(small.poll());
            }
        }

        public double findMedian() {
            if (small.size() > large.size()) {
                return small.peek();
            } else if (small.size() < large.size()) {
                return large.peek();
            } else {
                return (small.peek() + large.peek()) / 2.0;
            }
        }
    }

    //[299].猜数字游戏
    public static String getHint(String secret, String guess) {
        int[] cache = new int[10];
        int bullCount = 0;
        for (int i = 0; i < secret.length(); i++) {
            if (secret.charAt(i) == guess.charAt(i)) {
                bullCount++;
            } else {
                cache[secret.charAt(i) - '0'] += 1;
            }
        }
        int cowCount = 0;
        for (int i = 0; i < guess.length(); i++) {
            char g = guess.charAt(i);
            if (secret.charAt(i) != g) {
                if (cache[g - '0'] > 0) {
                    cache[g - '0'] -= 1;
                    cowCount++;
                }
            }
        }
        return bullCount + "A" + cowCount + "B";
    }

    //[318].最大单词长度乘积
    public static int maxProduct(String[] words) {
        int size = words.length;
        if (size == 0) return 0;
        int[] values = new int[size];
        for (int i = 0; i < size; i++) {
            int r1 = 0;
            for (char ch : words[i].toCharArray()) {
                r1 |= 1 << ch - 'a';
            }
            values[i] = r1;
        }

        int res = 0;
        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {
                //不包含字母意味着 与等于0
                if ((values[i] & values[j]) == 0) {
                    res = Math.max(res, words[i].length() * words[j].length());
                }
            }
        }
        return res;
    }

    //[319].灯泡开关
    public static int bulbSwitch(int n) {
        int cnt = 0;
        for (int i = 1; i * i <= n; i++) {
            cnt++;
        }
        return cnt;
    }

    public class NestedIterator implements Iterator<Integer> {

        public class NestedInteger {

            // @return true if this NestedInteger holds a single integer, rather than a nested list.
            public boolean isInteger() {
                return true;
            }

            // @return the single integer that this NestedInteger holds, if it holds a single integer
            // Return null if this NestedInteger holds a nested list
            public Integer getInteger() {
                return 1;
            }

            // @return the nested list that this NestedInteger holds, if it holds a nested list
            // Return null if this NestedInteger holds a single integer
            public List<NestedInteger> getList() {
                return null;
            }
        }

        Deque<NestedInteger> stack = new ArrayDeque<>();

        public NestedIterator(List<NestedInteger> nestedList) {
            for (int i = nestedList.size() - 1; i >= 0; i--) {
                stack.push(nestedList.get(i));
            }
        }

        @Override
        public Integer next() {
            return stack.poll().getInteger();
        }

        @Override
        public boolean hasNext() {
            if (stack.isEmpty()) {
                return false;
            } else {
                if (!stack.peek().isInteger()) {
                    List<NestedInteger> list = stack.poll().getList();
                    for (int i = list.size() - 1; i >= 0; i--) {
                        stack.push(list.get(i));
                    }
                    //有可能还是一个嵌套，递归求解
                    return hasNext();
                } else {
                    return true;
                }
            }
        }
    }

    //[344].反转字符串
    public static void reverseString(char[] s) {
        if (s == null) return;
        int l = 0, r = s.length - 1;
        while (l < r) {
            char temp = s[l];
            s[l] = s[r];
            s[r] = temp;
            l++;
            r--;
        }
    }

    //[345].反转字符串中的元音字母
    public static String reverseVowels(String s) {
        if (s == null) return null;
        char[] res = s.toCharArray();
        int l = 0, r = res.length - 1;
        while (l < r) {
            if (!isVowel(res[l])) {
                l++;
                continue;
            }
            if (!isVowel(res[r])) {
                r--;
                continue;
            }
            //直到找到元音字母交换
            char temp = res[l];
            res[l] = res[r];
            res[r] = temp;
            l++;
            r--;
        }
        return String.valueOf(res);
    }

    private static boolean isVowel(char ch) {
        return ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u'
                || ch == 'A' || ch == 'E' || ch == 'I' || ch == 'O' || ch == 'U';
    }

    //[355].设计推特
    public static class Twitter {

        private class Tweet {
            private int tweetId;
            private long timestamp;
            private Tweet next;

            public Tweet(int tweetId, long timestamp) {
                this.tweetId = tweetId;
                this.timestamp = timestamp;
            }

            public void setNext(Tweet next) {
                this.next = next;
            }
        }

        Map<Integer, Tweet> userTweets = new HashMap<>();
        Map<Integer, Set<Integer>> followers = new HashMap<>();
        long timestamp = 0;

        /**
         * Initialize your data structure here.
         */
        public Twitter() {
        }

        /**
         * Compose a new tweet.
         */
        public void postTweet(int userId, int tweetId) {
            if (userTweets.containsKey(userId)) {
                Tweet head = userTweets.get(userId);
                Tweet latest = new Tweet(tweetId, timestamp++);
                latest.next = head;
                userTweets.put(userId, latest);
            } else {
                userTweets.put(userId, new Tweet(tweetId, timestamp++));
            }
        }

        /**
         * Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
         */
        public List<Integer> getNewsFeed(int userId) {
            Set<Integer> users = new HashSet<>();
            users.add(userId);
            Set<Integer> followees = followers.get(userId);
            if (followees != null) {
                users.addAll(followees);
            }

            List<Integer> res = new ArrayList<>();
            if (users.isEmpty()) {
                return res;
            }

            PriorityQueue<Tweet> queue = new PriorityQueue<>((a, b) -> (int) (b.timestamp - a.timestamp));
            for (int id : users) {
                if (!userTweets.containsKey(id)) {
                    continue;
                }
                queue.add(userTweets.get(id));
            }

            while (!queue.isEmpty()) {
                Tweet tweet = queue.poll();
                res.add(tweet.tweetId);
                if (res.size() == 10) {
                    return res;
                }

                if (tweet.next != null) {
                    queue.add(tweet.next);
                }
            }

            return res;
        }

        /**
         * Follower follows a followee. If the operation is invalid, it should be a no-op.
         */
        public void follow(int followerId, int followeeId) {
            followers.putIfAbsent(followerId, new HashSet<>());
            Set<Integer> followees = followers.get(followerId);
            followees.add(followeeId);
        }

        /**
         * Follower unfollows a followee. If the operation is invalid, it should be a no-op.
         */
        public void unfollow(int followerId, int followeeId) {
            Set<Integer> followees = followers.get(followerId);
            if (followees != null) {
                followees.remove(followeeId);
            }
        }
    }

    //[371].两整数之和
    public static int getSum(int a, int b) {
        while (b != 0) {
            int lower = a ^ b;
            int carry = (a & b & 0x7fffffff) << 1;
            b = carry;
            a = lower;
            System.out.println(a + "-" + b);
        }
        return a;
    }

    //[372].超级次方
    public static int superPow(int a, int[] b) {
        if (b.length == 0) return 1;
        int part1 = myPow(a, b[b.length - 1]);
        b = Arrays.copyOf(b, b.length - 1);
        int part2 = myPow(superPow(a, b), 10);
        return (part1 * part2) % 1337;
    }

    private static int myPow(int a, int b) {
        int base = 1337;
        if (b == 0) return 1;

        a %= base;
        if (b % 2 == 1) {
            return (myPow(a, b - 1) * a) % base;
        } else {
            int half = myPow(a, b / 2);
            return (half * half) % base;
        }
    }

    //[373].查找和最小的K对数字
    public static List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<List<Integer>> res = new LinkedList<>();
        if (nums1.length == 0 || nums2.length == 0) return res;
        //二元组中放的是坐标,小顶堆
        PriorityQueue<int[]> queue = new PriorityQueue<>(Comparator.comparingInt(o -> nums1[o[0]] + nums2[o[1]]));
        for (int i = 0; i < Math.min(nums1.length, k); i++) {
            queue.add(new int[]{i, 0});
        }

        while (k > 0 && !queue.isEmpty()) {
            int[] pair = queue.poll();
            res.add(Arrays.asList(nums1[pair[0]], nums2[pair[1]]));

            //等于就不需要添加了
            if (pair[1] + 1 < nums2.length) {
                queue.add(new int[]{pair[0], pair[1] + 1});
            }
            k--;
        }
        return res;
    }

    //[380].常数时间插入、删除和获取随机元素
    public static class RandomizedSet {
        Map<Integer, Integer> map;
        List<Integer> data;
        Random rand = new Random();

        /**
         * Initialize your data structure here.
         */
        public RandomizedSet() {
            this.map = new HashMap<>();
            this.data = new ArrayList<>();
        }

        /**
         * Inserts a value to the set. Returns true if the set did not already contain the specified element.
         */
        public boolean insert(int val) {
            if (map.containsKey(val)) {
                return false;
            }
            int index = data.size();
            map.put(val, index);
            data.add(index, val);
            return true;
        }

        /**
         * Removes a value from the set. Returns true if the set contained the specified element.
         */
        public boolean remove(int val) {
            if (!map.containsKey(val)) {
                return false;
            }
            int index = map.get(val);
            int lastIndex = data.size() - 1;
            int lastValue = data.get(lastIndex);

            data.set(index, lastValue);
            map.put(lastValue, index);

            data.remove(lastIndex);
            map.remove(val);
            return true;
        }

        /**
         * Get a random element from the set.
         */
        public int getRandom() {
            return data.get(rand.nextInt(data.size()));
        }
    }

    //[390].消除游戏
    public static int lastRemaining(int n) {
        return calForLastRemaining(n);
    }

    private static int calForLastRemaining(int n) {
        if (n == 1) return 1;
        if (n == 2) return 2;

        if (n % 2 != 0) {
            //dp[9] = dp[8]
            return calForLastRemaining(n - 1);
        } else {
            // 逆序等于 6, 正序等于4 (2 + 8 -6)
            // dp[8] = 2 4 6 8
            // dp[4] = 1 2 3 4
            // dp[8] = 2(1 + 4 - dp[4])
            return 2 * (1 + n / 2 - calForLastRemaining(n / 2));
        }
    }

    //[391].完美矩形
    public static boolean isRectangleCover(int[][] rectangles) {
        int X1 = Integer.MAX_VALUE, Y1 = Integer.MAX_VALUE, X2 = Integer.MIN_VALUE, Y2 = Integer.MIN_VALUE;
        int totalArea = 0;
        //只保留度为奇数的顶点，即为大矩形的顶点
        Set<String> oddPoints = new HashSet<>();
        for (int[] rectangle : rectangles) {
            int x1 = rectangle[0], y1 = rectangle[1], x2 = rectangle[2], y2 = rectangle[3];

            List<String> realPoints = new ArrayList<>();
            realPoints.add(x1 + "," + y1);
            realPoints.add(x1 + "," + y2);
            realPoints.add(x2 + "," + y1);
            realPoints.add(x2 + "," + y2);

            for (String realPoint : realPoints) {
                if (oddPoints.contains(realPoint)) {
                    oddPoints.remove(realPoint);
                } else {
                    oddPoints.add(realPoint);
                }
            }

            totalArea += (x2 - x1) * (y2 - y1);
            X1 = Math.min(x1, X1);
            Y1 = Math.min(y1, Y1);

            X2 = Math.max(x2, X2);
            Y2 = Math.max(y2, Y2);
        }
        if (totalArea != (X2 - X1) * (Y2 - Y1)) {
            return false;
        }
        //比如有6个顶点，非矩形， 比如重叠，有多余的顶点，非矩形
        if (oddPoints.size() != 4) {
            return false;
        }

        if (!oddPoints.contains(X1 + "," + Y1)) {
            return false;
        }
        if (!oddPoints.contains(X1 + "," + Y2)) {
            return false;
        }
        if (!oddPoints.contains(X2 + "," + Y1)) {
            return false;
        }
        if (!oddPoints.contains(X2 + "," + Y2)) {
            return false;
        }
        return true;
    }

    //[392].判断子序列
    public static boolean isSubsequence(String s, String t) {
        if (s.length() == 0) return true;
        int i = 0, j = 0;
        while (j < t.length()) {
            if (s.charAt(i) == t.charAt(j)) {
                i++;
                if (i == s.length()) {
                    return true;
                }
            }
            j++;

        }
        return false;
    }

    //[393].UTF-8编码验证
    public static boolean validUtf8(int[] data) {
        //还需要几个10xxxxxx
        int count = 0;
        for (int i = 0; i < data.length; i++) {
            if (count > 0) {
                if (data[i] >> 6 != 0x02) {
                    return false;
                }
                count--;
            } else if (data[i] >> 3 == 0x1E) {
                count = 3;
            } else if (data[i] >> 4 == 0x0E) {
                count = 2;
            } else if (data[i] >> 5 == 0x06) {
                count = 1;
            } else if (data[i] >> 7 == 0x00) {
                count = 0;
            } else {
                //如果count==0, 10x开头，则返回false
                //如果开头不正确，则返回false
                return false;
            }
        }
        return count == 0;
    }

    //[397].整数替换
    public static int integerReplacement(int n) {
        if (n <= 1) return 0;
        if (n % 2 == 0) {
            return integerReplacement(n / 2) + 1;
        } else {
            //n-1 n+1取一半+2, 防止越界
            // 7 - 7/2 = 4
            // 7/2 = 3
            return Math.min(integerReplacement(n - n / 2), integerReplacement(n / 2)) + 2;
        }
    }

    //[398].随机数索引
    public static class Solution {
        private int[] arr;

        public Solution(int[] nums) {
            this.arr = nums;
        }

        public int pick(int target) {
            List<Integer> index = new ArrayList<>();
            for (int i = 0; i < arr.length; i++) {
                if (target == arr[i]) {
                    index.add(i);
                }
            }

            Random random = new Random();
            return index.get(random.nextInt(index.size()));
        }
    }

    //[400].第N个数字
    public static int findNthDigit(int n) {
        if (n < 10) return n;
        //i表示第几层索引，cnt表示每层个数9,90,900，length表示实际上一层的前缀总个数
        long i = 1, cnt = 9, length = 0;
        for (; length + cnt * i < n; i++) {
            length += cnt * i;
            cnt *= 10;
        }
        //实际的数字
        long num = (long) Math.pow(10, i - 1) + (n - length - 1) / i;
        //第几位
        int index = (int) ((n - length - 1) % i);
        return String.valueOf(num).charAt(index) - '0';
    }

    //[423].从英文中重建数字
    public static String originalDigits(String s) {
        //zero two four six eight three five seven nine one
        // z   w    u    x   g      h     f    s    i    n

        int[] count = new int[26];
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            count[ch - 'a']++;
        }

        int[] output = new int[10];
        output[0] = count['z' - 'a'];
        output[2] = count['w' - 'a'];
        output[4] = count['u' - 'a'];
        output[6] = count['x' - 'a'];
        output[8] = count['g' - 'a'];

        output[3] = count['h' - 'a'] - output[8];
        output[5] = count['f' - 'a'] - output[4];
        output[7] = count['s' - 'a'] - output[6];
        output[9] = count['i' - 'a'] - output[5] - output[8] - output[6];
        output[1] = count['n' - 'a'] - 2 * output[9] - output[7];

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i <= 9; i++) {
            for (int j = 0; j < output[i]; j++) {
                sb.append(i);
            }
        }
        return sb.toString();
    }

    //[451].根据字符出现频率排序
    public static String frequencySort(String s) {
        Map<Character, Integer> map = new HashMap<>();
        char[] chs = s.toCharArray();
        for (char ch : chs) {
            int count = map.getOrDefault(ch, 0);
            map.put(ch, ++count);
        }

        PriorityQueue<Character> queue = new PriorityQueue<>((o1, o2) -> map.get(o2) - map.get(o1));
        map.forEach((key, value) -> queue.add(key));

        StringBuilder sb = new StringBuilder();
        while (!queue.isEmpty()) {
            char ch = queue.poll();
            int feq = map.get(ch);
            for (int i = 0; i < feq; i++) {
                sb.append(ch);
            }
        }
        return sb.toString();
    }

    //[460].LFU缓存
    public static class LFUCache {
        private Map<Integer, Integer> keyToVal;
        private Map<Integer, Integer> keyToFre;
        private Map<Integer, LinkedHashSet<Integer>> freToKeys;
        private int cap;
        private int minFre;

        public LFUCache(int capacity) {
            keyToVal = new HashMap<>();
            keyToFre = new HashMap<>();
            freToKeys = new HashMap<>();
            cap = capacity;
            minFre = 0;
        }

        public int get(int key) {
            if (!keyToVal.containsKey(key)) return -1;
            increaseFre(key);
            return keyToVal.get(key);
        }

        public void put(int key, int value) {
            if (cap <= 0) return;
            if (keyToVal.containsKey(key)) {
                increaseFre(key);
                keyToVal.put(key, value);
                return;
            }

            //容量超了淘汰
            if (this.cap <= keyToVal.size()) {
                LinkedHashSet<Integer> list = freToKeys.get(minFre);
                int deleteKey = list.iterator().next();
                list.remove(deleteKey);
                if (list.isEmpty()) {
                    //不需要便跟minFre
                    freToKeys.remove(minFre);
                }
                keyToVal.remove(deleteKey);
                keyToFre.remove(deleteKey);
            }

            keyToVal.put(key, value);
            keyToFre.put(key, 1);
            freToKeys.putIfAbsent(1, new LinkedHashSet<>());
            freToKeys.get(1).add(key);
            minFre = 1;
        }

        private void increaseFre(int key) {
            int fre = keyToFre.get(key);
            keyToFre.put(key, fre + 1);
            freToKeys.putIfAbsent(fre + 1, new LinkedHashSet<>());
            freToKeys.get(fre + 1).add(key);

            LinkedHashSet<Integer> list = freToKeys.get(fre);
            list.remove(key);
            if (list.isEmpty()) {
                freToKeys.remove(fre);
                if (fre == minFre) {
                    minFre++;
                }
            }
        }
    }

    //[468].验证IP地址
    public static String validIPAddress(String IP) {
        if (IP.contains(".")) {
            String[] ipv4 = IP.split("\\.", -1);
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
                if (num < 0 || num > 255 || String.valueOf(num).length() != ip.length()) {
                    return "Neither";
                }
            }
            return "IPv4";
        } else if (IP.contains(":")) {
            String[] ipv6 = IP.split(":", -1);
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
        } else {
            return "Neither";
        }
    }

    //[478].在圆内随机生成点
    public class Solution478 {

        private double radius;
        private double xCenter;
        private double yCenter;

        public Solution478(double radius, double x_center, double y_center) {
            this.radius = radius;
            this.xCenter = x_center;
            this.yCenter = y_center;
        }

        public double[] randPoint() {
            double r = radius * Math.sqrt(Math.random());
            double angle = Math.random() * 2 * Math.PI;
            return new double[]{r * Math.cos(angle) + xCenter, r * Math.sin(angle) + yCenter};
        }
    }

    //[481].神奇字符串
    public static int magicalString(int n) {
        StringBuilder sb = new StringBuilder();
        sb.append(1);
        //频次指针
        int index = 1;
        while (sb.length() < n) {
            //频次字符串不够，需要根据前一个字符进行生成
            if (index == sb.length()) {
                sb.append(sb.charAt(sb.length() - 1) == '1' ? 22 : 1);
                index++;
            } else {
                //最后一个字符是1，下一个字符是2
                if (sb.charAt(sb.length() - 1) == '1') {
                    sb.append(sb.charAt(index++) == '1' ? 2 : 22);
                } else {
                    //最后一个字符是2，下一组字符是1
                    sb.append(sb.charAt(index++) == '1' ? 1 : 11);
                }
            }
        }

        int count = 0;
        for (int i = 0; i < n; i++) {
            count += sb.charAt(i) == '1' ? 1 : 0;
        }
        return count;
    }

    //[497].非重叠矩形中的随机点
    public static class Solution497 {

        Random random = new Random();
        int totalArea;
        TreeMap<Integer, int[]> map = new TreeMap<>();

        public Solution497(int[][] rects) {
            for (int[] rect : rects) {
                int area = (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1);
                totalArea += area;
                map.put(totalArea, rect);
            }
        }

        public int[] pick() {
            int randomArea = random.nextInt(totalArea) + 1;
            int ceilingArea = map.ceilingKey(randomArea);
            int[] rect = map.get(ceilingArea);
            int width = rect[2] - rect[0] + 1;
            int offset = ceilingArea - randomArea;
            return new int[]{rect[0] + offset % width, rect[1] + offset / width};
        }
    }

    //[519].随机翻转矩阵
    public static class Solution519 {

        int row, col, n;
        Map<Integer, Integer> map;
        Random random;

        public Solution519(int n_rows, int n_cols) {
            row = n_rows;
            col = n_cols;
            n = row * col;
            map = new HashMap<>();
            random = new Random();
        }

        public int[] flip() {
            if (n < 0) return null;
            int r = random.nextInt(n--);
            int x = map.getOrDefault(r, r);
            //保证取到重复位置的时候，实际位置是不一样的
            map.put(r, map.getOrDefault(n, n));
            return new int[]{x / col, x % col};
        }

        public void reset() {
            map.clear();
            n = row * col;
        }

    }

    //[524].通过删除字母匹配到字典里最长单词
    public static String findLongestWord(String s, List<String> d) {
        Collections.sort(d, (a, b) -> a.length() == b.length() ? a.compareTo(b) : b.length() - a.length());
        for (String dict : d) {
            if (isSubSequence(dict, s)) {
                return dict;
            }
        }
        return "";
    }

    //x是否是y的子序列
    private static boolean isSubSequence(String x, String y) {
        int i = 0, j = 0;
        for (; i < x.length() && j < y.length(); j++) {
            if (x.charAt(i) == y.charAt(j)) {
                i++;
            }
        }
        return i == x.length();

    }

    //[528].按权重随机选择
    public static class Solution528 {

        TreeMap<Integer, Integer> map = new TreeMap<>();
        Random random = new Random();
        int sum = 0;

        public Solution528(int[] w) {
            for (int i = 0; i < w.length; i++) {
                map.put(sum, i);
                sum += w[i];
            }
        }

        public int pickIndex() {
            int r = random.nextInt(sum);
            int floorKey = map.floorKey(r);
            return map.get(floorKey);
        }
    }

    //[535].TinyURL的加密与解密
    public static class Codec535 {

        String encode = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        Map<String, String> shortToLong = new HashMap<>();
        Random random = new Random();

        // Encodes a URL to a shortened URL.
        public String encode(String longUrl) {
            String key = getKey();
            while (shortToLong.containsKey(key)) {
                key = getKey();
            }

            shortToLong.put(key, longUrl);
            return "http://tinyurl.com/" + key;
        }

        // Decodes a shortened URL to its original URL.
        public String decode(String shortUrl) {
            String key = shortUrl.replace("http://tinyurl.com/", "");
            return shortToLong.get(key);
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
    public static String complexNumberMultiply(String a, String b) {
        String[] first = a.split("\\+");
        String[] second = b.split("\\+");

        int x = Integer.parseInt(first[0]);
        int y = Integer.parseInt(second[0]);
        int w = Integer.parseInt(first[1].replace("i", ""));
        int z = Integer.parseInt(second[1].replaceAll("i", ""));

        int real = x * y - w * z;
        int complex = y * w + x * z;
        return "" + real + "+" + complex + "i";
    }

    //[554].砖墙
    public static int leastBricks(List<List<Integer>> wall) {
        Map<Integer, Integer> positionCount = new HashMap<>();
        int maxCount = 0;
        for (List<Integer> bricks : wall) {
            int sum = 0;
            for (int i = 0; i < bricks.size() - 1; i++) {
                sum += bricks.get(i);
                positionCount.put(sum, positionCount.getOrDefault(sum, 0) + 1);
                maxCount = Math.max(maxCount, positionCount.get(sum));
            }
        }
        return wall.size() - maxCount;
    }

    //[553].最优除法
    public static String optimalDivision(int[] nums) {
        int n = nums.length;
        if (n == 0) return "";
        if (n == 1) return "" + nums[0];
        if (n == 2) return nums[0] + "/" + nums[1];

        StringBuilder sb = new StringBuilder();
        sb.append(nums[0]).append("/(").append(nums[1]);
        for (int i = 2; i < n; i++) {
            sb.append("/" + nums[i]);
        }
        sb.append(")");
        return sb.toString();
    }

    //[592].分数加减运算
    public static String fractionAddition(String expression) {
        expression = expression.replaceAll("-", "+-");
        String[] strs = expression.split("\\+");

        int sumFenmu = 1;
        for (String str : strs) {
            if (str.length() == 0) {
                continue;
            }
            int fenmu = Integer.parseInt(str.split("/")[1]);
            sumFenmu *= fenmu;
        }
        int sumFenzi = 0;
        for (String str : strs) {
            if (str.length() == 0) {
                continue;
            }
            int fenzi = Integer.parseInt(str.split("/")[0]);
            int fenmu = Integer.parseInt(str.split("/")[1]);
            sumFenzi += sumFenmu / fenmu * fenzi;
        }
        int yueshu = gcd(sumFenzi, sumFenmu);
        return sumFenzi / yueshu + "/" + sumFenmu / yueshu;
    }

    //求两个数的最大公约数
    public static int gcd(int m, int n) {
        m = Math.abs(m);
        n = Math.abs(n);
        int result = 0;
        while (n != 0) {
            result = m % n;
            m = n;
            n = result;
        }
        return m;
    }

    //[593].有效的正方形
    public static boolean validSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
        int[][] ps = new int[][]{p1, p2, p3, p4};
        Arrays.sort(ps, (a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
        //四边相等，且对角线相等
        return calculateDistance(ps[0], ps[1]) != 0
                && calculateDistance(ps[0], ps[1]) == calculateDistance(ps[1], ps[3])
                && calculateDistance(ps[1], ps[3]) == calculateDistance(ps[3], ps[2])
                && calculateDistance(ps[3], ps[2]) == calculateDistance(ps[2], ps[0])
                && calculateDistance(ps[0], ps[3]) == calculateDistance(ps[1], ps[2]);
    }

    private static int calculateDistance(int[] p, int[] q) {
        return (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]);
    }

    //[804].唯一摩斯密码词
    public static int uniqueMorseRepresentations(String[] words) {
        String[] morse = new String[]{".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--", "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."};
        HashSet<String> set = new HashSet<>();
        for (String word : words) {
            char[] each = word.toCharArray();
            StringBuilder sb = new StringBuilder();
            for (char ch : each) {
                sb.append(morse[ch - 'a']);
            }
            set.add(sb.toString());
        }
        return set.size();
    }

    public static void main(String[] args) {
//        [7].z字形变换
//        System.out.println(convert("LEETCODEISHIRING", 4));
//        System.out.println(convert("", 4));
//
//        [11].盛最多水的容器
//        System.out.println(maxArea(new int[]{4,3,2,1,4}));
//        System.out.println(maxArea(new int[]{1,1}));
//        System.out.println(maxArea(new int[]{1,2,1}));
//
//        [28]. 实现strStr()
//        System.out.println(strStr("abaclallb", "ll"));
//        System.out.println(strStr("aaaaa", "bba"));
//        System.out.println(strStr("hello", "ll"));
//        System.out.println(strStr("", "1"));
//        System.out.println(strStr("aaaaaab", "aab"));
//
//        [42].接雨水
//        System.out.println(trap(new int[]{4,2,0,3,2,5}));
//
//        [43].字符串相乘
//        System.out.println(multiply("123", "456"));
//        System.out.println(multiply("2", "3"));
//        System.out.println(multiply("123", "89"));
//
//        [49].字母异位分组
//        System.out.println(groupAnagrams(new String[]{"eat", "tea", "tan", "ate", "nat", "bat"}));
//        System.out.println(groupAnagrams(new String[]{}));
//
//        [130].被围绕的区域
//        char[][] board = new char[][]{{'X', 'X', 'X', 'X'}, {'X', 'O', 'O', 'X'}, {'X', 'X', 'O', 'X'}, {'X', 'O', 'X', 'X'}};
//        char[][] board2 = new char[][]{{'O'}};
//        solve(board);
//
//        [133].克隆图
//        Node f1 = new Node(1);
//        Node f2 = new Node(2);
//        Node f3 = new Node(3);
//        Node f4 = new Node(4);
//        f1.neighbors = Arrays.asList(f2, f4);
//        f2.neighbors = Arrays.asList(f1, f3);
//        f3.neighbors = Arrays.asList(f2, f4);
//        f4.neighbors = Arrays.asList(f1, f3);
//        Node res = cloneGraph(f1);
//
//        [137]只出现一次的数字II
//        System.out.println(singleNumber2(new int[]{0,1,0,1,0,1,99}));
//        System.out.println(singleNumber2(new int[]{99}));
//
//        [138].复制带随机指针的链表
//        Node o = new Node(7);
//        Node t = new Node(13);
//        Node t1 = new Node(11);
//        Node f = new Node(10);
//        Node s = new Node(1);
//        o.next = t;
//        t.next = t1;
//        t1.next = f;
//        f.next = s;
//        t.random = o;
//        t1.random = s;
//        f.random = t1;
//        s.random = o;
//        Node x = copyRandomList(o);
//
//        [151]翻转字符串里的单词
//        System.out.println(reverseWords("a good   example"));
//        System.out.println(reverseWords("  hello world!  "));
//
//        [146].LRU缓存机制
//        LRUCache cache = new LRUCache(2 /* 缓存容量 */);
//        cache.put(1, 1);
//        cache.put(2, 2);
//        System.out.println(cache.get(1));       // 返回  1
//        cache.put(3, 3);    // 该操作会使得关键字 2 作废
//        System.out.println(cache.get(2));       // 返回 -1 (未找到)
//        cache.put(4, 4);    // 该操作会使得关键字 1 作废
//        System.out.println(cache.get(1));       // 返回 -1 (未找到)
//        System.out.println(cache.get(3));       // 返回  3
//        System.out.println(cache.get(4));       // 返回  4
//
//        [165].比较版本号
//        System.out.println(compareVersion("0.1", "1.1"));
//        System.out.println(compareVersion("1.0.1", "1"));
//        System.out.println(compareVersion("1.01", "1.001"));
//
//        [166].分数到小数
//        System.out.println(fractionToDecimal(-4, 17));
//        System.out.println(fractionToDecimal(2, 3));
//
//        [187].重复的DNA序列
//        System.out.println(findRepeatedDnaSequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"));
//        System.out.println(findRepeatedDnaSequences("AAAAAAAAAAAA"));
//        System.out.println(findRepeatedDnaSequences("AAAAAAAAAA"));
//
//        [200].岛屿数量
//        char[][] islands = {{'1', '1', '0', '0', '0'}, {'1', '1', '0', '0', '0'}, {'0', '0', '1', '0', '0'}, {'0', '0', '0', '1', '1'}};
//        char[][] islands2 = {{'1', '1', '1', '1', '0'}, {'1', '1', '0', '1', '0'}, {'1', '1', '0', '0', '0'}, {'0', '0', '0', '0', '0'}};
//        System.out.println(numIslands(islands2));
//
//        [201].数字范围按位与
//        System.out.println(rangeBitwiseAnd(1,1));
//        System.out.println(rangeBitwiseAnd(7, 8));
//        System.out.println(rangeBitwiseAnd(5,7));
//
//        [210].课程表II
//        System.out.println(Arrays.toString(findOrder(4, new int[][]{{1, 0}, {2, 0}, {3, 1}, {3, 2}})));
//        System.out.println(Arrays.toString(findOrder(2, new int[][]{{1, 0}, {0, 1}})));
//
//        [223].矩形面积
//        System.out.println(computeArea(-3, 0, 3, 4, 0, -1, 9, 2));
//
//        [241].为运算表达式设计优先级
//        System.out.println(diffWaysToCompute("2*3-14*5"));
//        System.out.println(diffWaysToCompute("2-1-1"));
//
//        [263].丑数
//        System.out.println(isUgly(6));
//        System.out.println(isUgly(8));
//        System.out.println(isUgly(14));
//        System.out.println(isUgly(2123366400));
//
//        [264].丑数II
//        System.out.println(nthUglyNumber(10));
//        System.out.println(nthUglyNumber(1690));
//
//        [260].只出现一次的数字III
//        singleNumber3(new int[]{1, 2, 1, 3, 2, 5});
//
//        [295].数据流的中位数
//        MedianFinder finder = new MedianFinder();
//        finder.addNum(1);
//        finder.addNum(2);
//        System.out.println(finder.findMedian());
//        finder.addNum(3);
//        System.out.println(finder.findMedian());
//
//        [299].猜数字游戏
//        System.out.println(getHint("1807", "7810"));
//        System.out.println(getHint("1123", "0111"));
//
//        [318].最大单词长度乘积
//        System.out.println(maxProduct(new String[]{"abcw", "baz", "foo", "bar", "xtfn", "abcdef"}));
//        System.out.println(maxProduct(new String[]{"a", "ab", "abc", "d", "cd", "bcd", "abcd"}));
//        System.out.println(maxProduct(new String[]{"a", "aa", "aaa", "aaaa"}));
//
//        [319].灯泡开关
//        System.out.println(bulbSwitch(12));
//
//        [344].反转字符串
//        char[] res = new char[]{};
//        reverseString(res);
//
//        [345].反转字符串中的元音字母
//        System.out.println(reverseVowels("leetcodo"));
//
//        [355].设计推特
//        Twitter twitter = new Twitter();
//        // 用户1发送了一条新推文 (用户id = 1, 推文id = 5).
//        twitter.postTweet(1, 5);
//        // 用户1的获取推文应当返回一个列表，其中包含一个id为5的推文.
//        System.out.println(twitter.getNewsFeed(1));
//        // 用户1关注了用户2.
//        twitter.follow(1, 2);
//        // 用户2发送了一个新推文 (推文id = 6).
//        twitter.postTweet(2, 6);
//        // 用户1的获取推文应当返回一个列表，其中包含两个推文，id分别为 -> [6, 5].
//        System.out.println(twitter.getNewsFeed(1));
//        twitter.unfollow(1, 2);
//        // 用户1的获取推文应当返回一个列表，其中包含一个id为5的推文.
//        System.out.println(twitter.getNewsFeed(1));
//
//        [371].两整数之和
//        System.out.println(getSum(2, 3));
//        System.out.println(getSum(-2, 3));
//
//        [372].超级次方
//        System.out.println(superPow(2147483647, new int[]{2, 0, 0}));
//        System.out.println(superPow(1, new int[]{4, 3, 3, 8, 5, 2}));
//        System.out.println(superPow(2, new int[]{3}));
//        System.out.println(superPow(2, new int[]{1, 0}));
//
//        [373].查找和最小的K对数字
//        System.out.println(kSmallestPairs(new int[]{1,1,2}, new int[]{1,2,3}, 2));
//        System.out.println(kSmallestPairs(new int[]{1, 7, 11}, new int[]{2, 4, 6}, 10));
//
//        [380].常数时间插入、删除和获取随机元素
//        RandomizedSet set = new RandomizedSet();
//        set.insert(1);
//        set.insert(2);
//        System.out.println(set.insert(2));;
//        set.insert(3);
//        set.remove(2);
//        System.out.println(set.getRandom());
//
//        [390].消除游戏
//        System.out.println(lastRemaining(10));
//
//        [391].完美矩形
//        System.out.println(isRectangleCover(new int[][]{{1,1,3,3},{3,1,4,2},{3,2,4,4},{1,3,2,4},{2,3,3,4}}));
//        System.out.println(isRectangleCover(new int[][]{{1,1,2,3},{1,3,2,4},{3,1,4,2},{3,2,4,4}}));
//        System.out.println(isRectangleCover(new int[][]{{1,1,3,3},{3,1,4,2},{1,3,2,4},{3,2,4,4}}));
//
//        [392].判断子序列
//        System.out.println(isSubsequence("axc", "ahbgdc"));
//        System.out.println(isSubsequence("abc", "ahbgdc"));
//
//        [393].UTF-8编码验证
//        System.out.println(validUtf8(new int[]{235, 140, 4}));
//        System.out.println(validUtf8(new int[]{240, 162, 138, 147, 145}));
//        System.out.println(validUtf8(new int[]{197, 130, 1}));
//
//        [397].整数替换
//        System.out.println(integerReplacement(7));
//        System.out.println(integerReplacement(Integer.MAX_VALUE));
//
//        [398].随机数索引
//        Solution solution = new Solution(new int[] {1,2,3,3,3});
//        System.out.println(solution.pick(3));
//
//        [400].第N个数字
//        System.out.println(findNthDigit(11));
//        System.out.println(findNthDigit(193));
//        System.out.println(findNthDigit(Integer.MAX_VALUE));
//
//        [423].从英文中重建数字
//        System.out.println(originalDigits("onetwothreethreefivefour"));
//
//        [451].根据字符出现频率排序
//        System.out.println(frequencySort("Aabb"));
//        System.out.println(frequencySort("tree"));
//        System.out.println(frequencySort("cccaaa"));
//
//        [460].LFU缓存
//        LFUCache lFUCache = new LFUCache(2);
//        lFUCache.put(1, 1);   // cache=[1,_], cnt(1)=1
//        lFUCache.put(2, 2);   // cache=[2,1], cnt(2)=1, cnt(1)=1
//        System.out.println(lFUCache.get(1));      // 返回 1
//        // cache=[1,2], cnt(2)=1, cnt(1)=2
//        lFUCache.put(3, 3);   // 去除键 2 ，因为 cnt(2)=1 ，使用计数最小
//        // cache=[3,1], cnt(3)=1, cnt(1)=2
//        System.out.println(lFUCache.get(2));      // 返回 -1（未找到）
//        System.out.println(lFUCache.get(3));    // 返回 3
//        // cache=[3,1], cnt(3)=2, cnt(1)=2
//        lFUCache.put(4, 4);   // 去除键 1 ，1 和 3 的 cnt 相同，但 1 最久未使用
//        // cache=[4,3], cnt(4)=1, cnt(3)=2
//        System.out.println(lFUCache.get(1));      // 返回 -1（未找到）
//        System.out.println(lFUCache.get(3));      // 返回 3
//        // cache=[3,4], cnt(4)=1, cnt(3)=3
//        System.out.println(lFUCache.get(4));      // 返回 4
//
//        [468].验证IP地址
//        System.out.println(validIPAddress("2001:0db8:85a3:0:0:8A2E:0370:7334"));
//        System.out.println(validIPAddress("2001:0db8:85a3:0:0:8A2E:0370:7334:"));
//        System.out.println(validIPAddress("1e1.4.5.6"));
//        System.out.println(validIPAddress("172.16.254.1"));
//        System.out.println(validIPAddress("1.0.1."));
//
//        [481].神奇字符串
//        System.out.println(magicalString(6));
//
//        [519].随机翻转矩阵
//        System.out.println(Arrays.toString(new Solution519(2, 3).flip()));
//
//        [524].通过删除字母匹配到字典里最长单词
//        System.out.println(findLongestWord("abpcplea", Arrays.asList("ale", "apple", "monkey", "plea")));
//        System.out.println(findLongestWord("abpcplea", Arrays.asList("a", "b", "c", "d")));
//
//        [528].按权重随机选择
//        Solution528 solution528 = new Solution528(new int[]{1, 3});
//        System.out.println(solution528.pickIndex());
//        System.out.println(solution528.pickIndex());
//        System.out.println(solution528.pickIndex());
//        System.out.println(solution528.pickIndex());
//        System.out.println(solution528.pickIndex());
//
//        [535].TinyURL的加密与解密
//        Codec535 codec535 = new Codec535();
//        String shortUrl = codec535.encode("https://leetcode.com/problems/535");
//        System.out.println(shortUrl);
//        System.out.println(codec535.decode(shortUrl));
//
//        [537].复数乘法
//        System.out.println(complexNumberMultiply("1+1i", "1+1i"));
//        System.out.println(complexNumberMultiply("1+-1i", "1+-1i"));
//
//        [553].最优除法
//        System.out.println(optimalDivision(new int[]{1000,100,10,2}));
//
        System.out.println(leastBricks(Arrays.asList(Arrays.asList(1, 2, 2, 1), Arrays.asList(3, 1, 2), Arrays.asList(1, 3, 2), Arrays.asList(2, 4), Arrays.asList(3, 1, 2), Arrays.asList(1, 3, 1, 1))));
        System.out.println(leastBricks(Arrays.asList(Arrays.asList(1), Arrays.asList(1), Arrays.asList(1))));
//        [592].分数加减运算
//        System.out.println(fractionAddition("-1/2+1/2"));
//        System.out.println(fractionAddition("-1/2+1/2+1/3"));
//        System.out.println(fractionAddition("1/3-1/2"));
//        System.out.println(fractionAddition("5/3+1/3"));
//        System.out.println(gcd(2, 3));
//
//        [593].有效的正方形
//        System.out.println(validSquare(new int[]{0, 0}, new int[]{1, 1}, new int[]{1, 0}, new int[]{0, 1}));
//
//        [804].唯一摩斯密码词
//        System.out.println(uniqueMorseRepresentations(new String[]{"gin", "zen", "gig", "msg"}));
    }

}
