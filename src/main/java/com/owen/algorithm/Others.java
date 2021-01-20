package com.owen.algorithm;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

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

    //[3]无重复子串的最长子串（双指针）
    public static int lengthOfLongestSubstring(String s) {
        //滑动窗口
        int[] position = new int[128];
        int start = 0, end = 0;
        int result = 0;
        char[] array = s.toCharArray();
        while (end < array.length) {
            char character = array[end];
            int lastMaxIndex = position[character];
            //滑动窗口缩小
            start = Math.max(start, lastMaxIndex);
            //更新最长子串的长度
            result = Math.max(result, end - start + 1);
            //更新字符的最大位置
            position[character] = end + 1;
            //滑动窗口扩大
            end++;
        }
        return result;
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
        int left = 0, right = height.length-1;
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

    //单调栈
    public static int[] nextGreaterNumber(int[] data) {
        int[] res = new int[data.length];
        Stack<Integer> stack = new Stack<>();
        for (int i = data.length - 1; i >= 0; i--) {
            while (!stack.empty() && data[i] >= stack.peek()) {
                //不断地剔除最小值，保留栈顶最大值
                stack.pop();
            }

            int max = stack.empty() ? -1 : stack.peek();
            res[i] = max;

            //放入每个值
            stack.push(data[i]);
        }
        return res;
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

    //[150].逆波兰表达式求值
    public static int evalRPN(String[] tokens) {
        Stack<String> stack = new Stack<>();
        for (String token : tokens) {
            if (token.equals("+") || token.equals("-") || token.equals("*") || token.equals("/")) {
                Integer lastNumber = Integer.parseInt(stack.pop());
                Integer firstNumber = Integer.parseInt(stack.pop());
                if (token.equals("+")) {
                    stack.push("" + (firstNumber + lastNumber));
                } else if (token.equals("-")) {
                    stack.push("" + (firstNumber - lastNumber));
                } else if (token.equals("*")) {
                    stack.push("" + (firstNumber * lastNumber));
                } else {
                    stack.push("" + (firstNumber / lastNumber));
                }
            } else {
                stack.push(token);
            }
        }
        return stack.isEmpty() ? 0 : Integer.parseInt(stack.pop());
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

    //[224].基本计算器
    public static int calculate(String s) {
        return dfsForCalculate(s.trim().toCharArray(), new AtomicInteger());
    }

    private static int dfsForCalculate(char[] chars, AtomicInteger index) {
        Stack<Integer> stack = new Stack<>();
        int num = 0;
        char sign = '+';
        for (; index.get() < chars.length; index.incrementAndGet()) {
            char ch = chars[index.get()];
            if (ch == ' ') {
                continue;
            }

            boolean isDig = '0' <= ch && ch <= '9';
            if (isDig) {
                num = num * 10 + (ch - '0');
            }

            //递归计算
            if (ch == '(') {
                index.incrementAndGet();
                num = dfsForCalculate(chars, index);
            }

            if (!isDig || index.get() == chars.length - 1) {
                switch (sign) {
                    case '+':
                        stack.push(num);
                        break;
                    case '-':
                        stack.push(-num);
                        break;
                }
                sign = ch;
                num = 0;
            }

            //递归结束计算
            if (ch == ')') {
                break;
            }
        }

        int res = 0;
        while (!stack.isEmpty()) {
            res += stack.pop();
        }
        return res;
    }

    //[225].队列实现栈
    class MyStack {

        Queue<Integer> queue;

        /**
         * Initialize your data structure here.
         */
        public MyStack() {
            queue = new LinkedList<>();
        }

        /**
         * Push element x onto stack.
         */
        public void push(int x) {
            int size = queue.size();
            queue.offer(x);
            for (int i = 0; i < size; i++) {
                queue.offer(queue.poll());
            }
        }

        /**
         * Removes the element on top of the stack and returns that element.
         */
        public int pop() {
            return queue.poll();
        }

        /**
         * Get the top element.
         */
        public int top() {
            return queue.peek();
        }

        /**
         * Returns whether the stack is empty.
         */
        public boolean empty() {
            return queue.isEmpty();
        }
    }

    //[227].基本计算器II
    public static int calculate2(String s) {
        char[] arr = s.trim().toCharArray();
        Stack<Integer> stack = new Stack<>();
        int num = 0;
        char preSign = '+';
        for (int i = 0; i < arr.length; i++) {
            char single = arr[i];
            if (single == ' ') {
                continue;
            }
            boolean isDig = '0' <= single && single <= '9';
            if (isDig) {
                num = num * 10 + (single - '0');
            }

            if (!isDig || i == arr.length - 1) {
                switch (preSign) {
                    case '+':
                        stack.push(num);
                        break;
                    case '-':
                        stack.push(-num);
                        break;
                    case '*':
                        stack.push(stack.pop() * num);
                        break;
                    case '/':
                        stack.push(stack.pop() / num);
                        break;
                }
                preSign = single;
                num = 0;
            }
        }

        int res = 0;
        while (!stack.isEmpty()) {
            res += stack.pop();
        }
        return res;
    }

    //[232].用栈实现队列
    class MyQueue {

        Stack<Integer> s1;
        Stack<Integer> s2;

        /**
         * Initialize your data structure here.
         */
        public MyQueue() {
            s1 = new Stack<>();
            s2 = new Stack<>();
        }

        /**
         * Push element x to the back of queue.
         */
        public void push(int x) {
            while (!s1.isEmpty()) {
                s2.push(s1.pop());
            }
            s1.push(x);
            while (!s2.isEmpty()) {
                s1.push(s2.pop());
            }
        }

        /**
         * Removes the element from in front of queue and returns that element.
         */
        public int pop() {
            return s1.pop();
        }

        /**
         * Get the front element.
         */
        public int peek() {
            return s1.peek();
        }

        /**
         * Returns whether the queue is empty.
         */
        public boolean empty() {
            return s1.isEmpty();
        }
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

    //[316].去除重复字母
    public static String removeDuplicateLetters(String s) {
        if (s == null) return null;
        int[] count = new int[256];
        for (char ch : s.toCharArray()) {
            count[ch]++;
        }
        Stack<Character> stack = new Stack<>();
        boolean[] inStack = new boolean[256];
        for (char ch : s.toCharArray()) {
            count[ch]--;
            if (inStack[ch]) {
                continue;
            }
            while (!stack.isEmpty() && stack.peek() > ch) {
                if (count[stack.peek()] == 0) {
                    break;
                }
                inStack[stack.pop()] = false;
            }

            stack.push(ch);
            inStack[ch] = true;
        }

        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            sb.append(stack.pop());
        }
        return sb.reverse().toString();
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

    //[331].验证二叉树的前序序列化
    public static boolean isValidSerialization(String preorder) {
        String[] pre = preorder.split(",");
        Stack<String> stack = new Stack<>();
        for (String ch : pre) {
            if (ch.equals("#")) {
                while (!stack.isEmpty() && stack.peek().equals("#")) {
                    stack.pop();
                    //##， ###
                    if (stack.isEmpty() || stack.pop().equals("#")) return false;
                }
                stack.push(ch);
            } else {
                stack.push(ch);
            }
        }
        //#要比正常节点数多1，最后剩下的一定是#
        return stack.size() == 1 && stack.peek().equals("#");
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

        /** Initialize your data structure here. */
        public Twitter() {
        }

        /** Compose a new tweet. */
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

        /** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
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

        /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
        public void follow(int followerId, int followeeId) {
            followers.putIfAbsent(followerId, new HashSet<>());
            Set<Integer> followees = followers.get(followerId);
            followees.add(followeeId);
        }

        /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
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

    //[394].字符串解码
    public static String decodeString(String s) {
        char[] chars = s.toCharArray();
        Stack<String> strings = new Stack<>();
        Stack<Integer> numbers = new Stack<>();
        int curNum = 0;
        StringBuilder curSB = new StringBuilder();
        for (char ch : chars) {
            if (ch >= '0' && ch <= '9') {
                //继续拼接数字
                curNum = curNum * 10 + ch - '0';
                continue;
            } else if (ch == ']') {
                //出栈操作，计算数字和字符串
                int prevNum = numbers.pop();
                String prevStr = strings.pop();
                StringBuilder temp = new StringBuilder();
                for (int i = 0; i < prevNum; i++) {
                    temp.append(curSB);
                }
                //新当前 = 上一个 + 当前*倍数
                curSB = new StringBuilder(prevStr + temp);
            } else if (ch == '[') {
                //数字和字符都结束了
                numbers.push(curNum);
                strings.push(curSB.toString());

                curNum = 0;
                curSB = new StringBuilder();
            } else {
                //继续拼接字符
                curSB.append(ch);
            }
        }
        return curSB.toString();
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


    //[402].移掉K位数字
    public static String removeKdigits(String num, int k) {
        if (k == num.length()) return "0";
        Stack<Character> stack = new Stack<>();
        for (char ch : num.toCharArray()) {
            while (!stack.isEmpty() && stack.peek() > ch && k > 0) {
                stack.pop();
                k--;
            }
            //前导0，不添加到栈中
            if (stack.isEmpty() && ch == '0') {
                continue;
            }
            stack.push(ch);
        }
        while (k > 0) {
            stack.pop();
            k--;
        }

        if (stack.isEmpty()) {
            return "0";
        }

        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            sb.append(stack.pop());
        }
        return sb.reverse().toString();
    }

    //[503].下一个更大的数II
    public static int[] nextGreaterElements(int[] nums) {
        int size = nums.length;
        int[] res = new int[size];
        Stack<Integer> stack = new Stack<>();
        for (int i = 2 * size - 1; i >= 0; i--) {
            while (!stack.empty() && stack.peek() <= nums[i % size]) {
                stack.pop();
            }
            res[i % size] = stack.empty() ? -1 : stack.peek();

            stack.push(nums[i % size]);
        }
        return res;
    }


    //[739].每日温度
    public static int[] dailyTemperatures(int[] T) {
        int size = T.length;
        int[] res = new int[size];
        Stack<Integer> stack = new Stack<>();
        for (int i = size - 1; i >= 0; i--) {
            while (!stack.isEmpty() && T[stack.peek()] <= T[i]) {
                stack.pop();
            }
            res[i] = stack.isEmpty() ? 0 : stack.peek() - i;
            stack.push(i);
        }
        return res;
    }

    //[895].最大频率栈
    static class FreqStack {
        private int maxFreq = 0;
        private Map<Integer, Stack<Integer>> freqNumbers = new HashMap<>();
        private Map<Integer, Integer> numberFreqs = new HashMap<>();

        public FreqStack() {
        }

        public void push(int x) {
            int feq = numberFreqs.getOrDefault(x, 0);
            feq++;
            numberFreqs.put(x, feq);

            freqNumbers.putIfAbsent(feq, new Stack<>());
            Stack<Integer> stack = freqNumbers.get(feq);
            stack.push(x);

            maxFreq = Math.max(maxFreq, feq);
        }

        public int pop() {
            Stack<Integer> stack = freqNumbers.get(maxFreq);
            int res = stack.pop();
            int freq = numberFreqs.get(res);
            freq--;
            numberFreqs.put(res, freq);
            if (stack.isEmpty()) {
                maxFreq--;
            }
            return res;
        }
    }

    public static void main(String[] args) {
//        [3]无重复子串的最长子串
//        System.out.println(lengthOfLongestSubstring("dvdf"));
//        System.out.println(lengthOfLongestSubstring("pwwkew"));
//        System.out.println(lengthOfLongestSubstring("tmmzuxt"));
//        System.out.println(lengthOfLongestSubstring("tmmzzuxt"));
//        System.out.println(lengthOfLongestSubstring("tmmzuuzt"));
//        System.out.println(lengthOfLongestSubstring("tmmzutt"));
//        System.out.println(lengthOfLongestSubstring("tmmzuzt"));
//        System.out.println(lengthOfLongestSubstring("tmmzuzuzt"));
//        System.out.println(lengthOfLongestSubstring("abcabcbb"));
//        System.out.println(lengthOfLongestSubstring("bbbbb"));
//        System.out.println(lengthOfLongestSubstring("abcabbcad"));
//        System.out.println(lengthOfLongestSubstring("aa"));
//        System.out.println(lengthOfLongestSubstring("pwwwwkeeew"));
//
//        [7]. z字形变换
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
//        单调栈
//        nextGreaterNumber(new int[]{1, 2, 3});
//        nextGreaterNumber(new int[]{2, 1, 2, 4, 3});
//

//        [130].被围绕的区域
//        char[][] board = new char[][]{{'X', 'X', 'X', 'X'}, {'X', 'O', 'O', 'X'}, {'X', 'X', 'O', 'X'}, {'X', 'O', 'X', 'X'}};
//        char[][] board2 = new char[][]{{'O'}};
//        solve(board);
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
//
//        [210].课程表II
//        System.out.println(Arrays.toString(findOrder(4, new int[][]{{1, 0}, {2, 0}, {3, 1}, {3, 2}})));
//        System.out.println(Arrays.toString(findOrder(2, new int[][]{{1, 0}, {0, 1}})));
//
//        [223].矩形面积
//        System.out.println(computeArea(-3, 0, 3, 4, 0, -1, 9, 2));
//
//        [224].基本计算器
//        System.out.println(calculate("(1+(4+5+2)-3)"));
//        System.out.println(calculate("(1+(4+5+2)-3)+(6+8)"));
//        System.out.println(calculate("(1+2"));
//
//        [227]基本计算器II
//        System.out.println(calculate2("3+2*2"));
//        System.out.println(calculate2(" 3+5 / 2 "));
//        System.out.println(calculate2(" 3/2 "));
//        System.out.println(calculate2(" 1 "));
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
//        [299].猜数字游戏
//        System.out.println(getHint("1807", "7810"));
//        System.out.println(getHint("1123", "0111"));
//
//        [316].去除重复字母
//        System.out.println(removeDuplicateLetters("cbacdcbc"));
//        System.out.println(removeDuplicateLetters("bcabc"));
//        System.out.println(removeDuplicateLetters(""));
//
//        [318].最大单词长度乘积
//        System.out.println(maxProduct(new String[]{"abcw", "baz", "foo", "bar", "xtfn", "abcdef"}));
//        System.out.println(maxProduct(new String[]{"a", "ab", "abc", "d", "cd", "bcd", "abcd"}));
//        System.out.println(maxProduct(new String[]{"a", "aa", "aaa", "aaaa"}));
//
//        [319].灯泡开关
//        System.out.println(bulbSwitch(12));
//
//        [331].验证二叉树的前序序列化
//        System.out.println(isValidSerialization("9,#,#,1"));
//        System.out.println(isValidSerialization("9,3,4,#,#,1,#,#,2,#,6,#,#"));
//
//        [344].反转字符串
//        char[] res = new char[]{};
//        reverseString(res);
//
//        [345].反转字符串中的元音字母
//        System.out.println(reverseVowels("leetcodo"));
//
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
//        [392].判断子序列
//        System.out.println(isSubsequence("axc", "ahbgdc"));
//        System.out.println(isSubsequence("abc", "ahbgdc"));
//
//        [393].UTF-8编码验证
//        System.out.println(validUtf8(new int[]{235, 140, 4}));
//        System.out.println(validUtf8(new int[]{240, 162, 138, 147, 145}));
//        System.out.println(validUtf8(new int[]{197, 130, 1}));
//
//        [394].字符串解码
//        System.out.println(decodeString("abc"));
//        System.out.println(decodeString("3[a]2[bc]"));
//        System.out.println(decodeString("3[a2[c]]"));
//        System.out.println(decodeString("2[abc]3[cd]ef"));
//        System.out.println(decodeString("abc3[cd]xyz"));
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
//        [402].移掉K位数字
//        System.out.println(removeKdigits("10200", 1));
//        System.out.println(removeKdigits("1432219", 3));
//        System.out.println(removeKdigits("10", 2));
//        System.out.println(removeKdigits("10016042692165669282207674747", 9));
//
//        [503].下一个更大元素II
//        System.out.println(Arrays.toString(nextGreaterElements(new int[]{1,2,1})));
//
//        [739].每日温度
//        System.out.println(Arrays.toString(dailyTemperatures(new int[]{73, 74, 75, 71, 69, 72, 76, 73})));
//
//        [895].最大频率栈
//        FreqStack freqStack = new FreqStack();
//        freqStack.push(5);
//        freqStack.push(7);
//        freqStack.push(5);
//        freqStack.push(7);
//        freqStack.push(4);
//        freqStack.push(5);
//        System.out.println(freqStack.pop());
//        System.out.println(freqStack.pop());
//        System.out.println(freqStack.pop());
//        System.out.println(freqStack.pop());

    }
}
