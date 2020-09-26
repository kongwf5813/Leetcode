package com.owen.algorithm;

import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicLong;

public class Others {

    public static class TreeNode {
        public int val;
        public TreeNode left;
        public TreeNode right;

        public TreeNode(int x) {
            val = x;
        }
    }

    public static class ListNode {
        public int val;
        public ListNode next;

        public ListNode() {
        }

        public ListNode(int val) {
            this.val = val;
        }

        public ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

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

    //28. 实现strStr()
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

    //[94] 二叉树的中序遍历(递归版本)
    public static List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        recursionForInorder(root, res);
        return res;
    }

    private static void recursionForInorder(TreeNode root, List<Integer> result) {
        if (root == null) {
            return;
        }
        recursionForInorder(root.left, result);
        result.add(root.val);
        recursionForInorder(root.right, result);
    }

    //[98].验证二叉搜索树
    public static boolean isValidBST(TreeNode root) {
        AtomicLong preMax = new AtomicLong(Long.MIN_VALUE);
        return dfsForBST(root, preMax);
    }

    private static boolean dfsForBST(TreeNode root, AtomicLong preMax) {
        if (root == null) return true;
        if (!dfsForBST(root.left, preMax)) {
            return false;
        }
        //访问当前节点：如果当前节点小于等于中序遍历的前一个节点
        if (root.val <= preMax.get()) {
            return false;
        }
        preMax.set(root.val);
        return dfsForBST(root.right, preMax);
    }

    //[94].二叉树的中序遍历(迭代版本)
    public static List<Integer> inorderTraversalV2(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        while (cur != null || !stack.isEmpty()) {
            //一直遍历访问左节点，压栈操作
            if (cur != null) {
                stack.push(cur);
                cur = cur.left;
            } else {
                //已经没有左节点了，把根返回，并且遍历右节点，压栈操作
                cur = stack.pop();
                res.add(cur.val);
                cur = cur.right;
            }
        }
        return res;
    }

    //[144].二叉树的前序遍历(迭代)
    public static List<Integer> preorderTraversalV2(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        List<Integer> res = new ArrayList<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            //先pop
            TreeNode top = stack.pop();
            res.add(top.val);
            if (top.right != null) {
                stack.push(top.right);
            }
            if (top.left != null) {
                stack.push(top.left);
            }
        }
        return res;
    }

    //[145].二叉树的后序遍历(迭代)
    public static List<Integer> postorderTraversalV2(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        while (cur != null || !stack.isEmpty()) {
            if (cur != null) {
                stack.push(cur);
                //先访问根
                res.add(0, cur.val);
                //再访问右
                cur = cur.right;
            } else {
                //最后访问左
                cur = stack.pop();
                cur = cur.left;
            }
        }
        return res;
    }

    //[144].二叉树的前序遍历(递归)
    public static List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        dfsForPreorder(root, res);
        return res;
    }

    private static void dfsForPreorder(TreeNode root, List<Integer> res) {
        if (root == null) {
            return;
        }
        res.add(root.val);
        dfsForPreorder(root.left, res);
        dfsForPreorder(root.right, res);
    }

    //[145].二叉树的后序遍历(递归)
    public static List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        dfsForPostorder(root, res);
        return res;
    }

    private static void dfsForPostorder(TreeNode root, List<Integer> res) {
        if (root == null) {
            return;
        }
        dfsForPostorder(root.left, res);
        dfsForPostorder(root.right, res);
        res.add(root.val);
    }

    //[100]相同的树
    public static boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null && q != null) return false;
        if (p != null && q == null) return false;
        if (p.val != q.val) return false;
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    //[101]对称二叉树
    public static boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return isSymmetric(root.left, root.right);
    }

    private static boolean isSymmetric(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
        if (left == null && right != null) return false;
        if (left != null && right == null) return false;
        if (left.val != right.val) return false;
        return isSymmetric(left.left, right.right) && isSymmetric(left.right, right.left);
    }

    //[102]二叉树的层序遍历
    public static List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedBlockingQueue<>();
        if (root != null) {
            queue.add(root);
        }
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> layer = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();
                if (cur.left != null) {
                    queue.add(cur.left);
                }
                if (cur.right != null) {
                    queue.add(cur.right);
                }
                layer.add(cur.val);
            }
            res.add(layer);
        }
        return res;
    }

    //[103].二叉树的锯齿形层次遍历
    public static List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        boolean flag = true;
        List<List<Integer>> res = new ArrayList<>();
        Stack<TreeNode> queue = new Stack<>();
        if (root != null) {
            queue.add(root);
        }
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> layer = new ArrayList<>();
            List<TreeNode> temp = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.pop();
                layer.add(cur.val);
                if (flag) {
                    if (cur.left != null) {
                        temp.add(cur.left);
                    }
                    if (cur.right != null) {
                        temp.add(cur.right);
                    }
                } else {
                    if (cur.right != null) {
                        temp.add(cur.right);
                    }
                    if (cur.left != null) {
                        temp.add(cur.left);
                    }
                }
            }
            for (TreeNode node : temp) {
                queue.add(node);
            }
            flag = !flag;
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

    //[105].从前序与中序遍历构造二叉树
    public static TreeNode buildTree(int[] preorder, int[] inorder) {
        int pLen = preorder.length;
        int iLen = inorder.length;
        if (pLen != iLen) return null;
        Map<Integer, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < iLen; i++) {
            indexMap.put(inorder[i], i);
        }
        return dfsForBuildTree(preorder, 0, pLen - 1, inorder, 0, iLen - 1, indexMap);
    }

    private static TreeNode dfsForBuildTree(int[] preorder, int ps, int pe, int[] indorder, int is, int ie, Map<Integer, Integer> indexMap) {
        if (ps > pe || is > ie) {
            return null;
        }
        int val = preorder[ps];
        TreeNode root = new TreeNode(val);
        int pIndex = indexMap.get(val);
        TreeNode left = dfsForBuildTree(preorder, ps + 1, pIndex - is + ps, indorder, is, pIndex - 1, indexMap);
        TreeNode right = dfsForBuildTree(preorder, pIndex - is + ps + 1, pe, indorder, pIndex + 1, ie, indexMap);
        root.left = left;
        root.right = right;
        return root;
    }

    //[106].从中序与后续遍历序列构造二叉树
    public static TreeNode buildTree2(int[] inorder, int[] postorder) {
        int iLen = inorder.length;
        int pLen = postorder.length;
        if (pLen != iLen) return null;
        Map<Integer, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < iLen; i++) {
            indexMap.put(inorder[i], i);
        }
        return dfsForBuildTree2(inorder, 0, iLen - 1, postorder, 0, pLen - 1, indexMap);
    }

    private static TreeNode dfsForBuildTree2(int[] inorder, int is, int ie, int[] postorder, int ps, int pe, Map<Integer, Integer> indexMap) {
        if (is > ie || ps > pe) return null;
        int val = postorder[pe];
        TreeNode root = new TreeNode(val);
        int pIndex = indexMap.get(val);
        TreeNode left = dfsForBuildTree2(inorder, is, pIndex - 1, postorder, ps, pIndex - 1 - is + ps, indexMap);
        TreeNode right = dfsForBuildTree2(inorder, pIndex + 1, ie, postorder, pIndex - is + ps, pe - 1, indexMap);
        root.left = left;
        root.right = right;
        return root;
    }

    //[107].二叉树的层次遍历II
    public static List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedBlockingQueue<>();
        if (root != null) {
            queue.add(root);
        }
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> layer = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();
                if (cur.left != null) {
                    queue.add(cur.left);
                }
                if (cur.right != null) {
                    queue.add(cur.right);
                }
                layer.add(cur.val);
            }
            res.add(0, layer);
        }
        return res;
    }

    //[108].将有序数组转换为二叉搜索树
    public static TreeNode sortedArrayToBST(int[] nums) {
        return dfsForSortedArray(nums, 0, nums.length - 1);
    }

    private static TreeNode dfsForSortedArray(int[] nums, int lo, int hi) {
        if (lo > hi) {
            return null;
        }
        int mid = lo + (hi - lo) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = dfsForSortedArray(nums, lo, mid - 1);
        root.right = dfsForSortedArray(nums, mid + 1, hi);
        return root;
    }

    //[109].有序链表转换为二叉搜索树
    public static TreeNode sortedListToBST(ListNode head) {
        if (head == null) return null;
        ListNode slow = head, fast = head, pre = null;
        while (fast != null && fast.next != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        TreeNode root = new TreeNode(slow.val);
        if (pre == null) return root;

        pre.next = null;
        root.left = sortedListToBST(head);
        root.right = sortedListToBST(slow.next);
        return root;
    }

    //[113].路径总和II
    public static List<List<Integer>> pathSum(TreeNode root, int sum) {
        LinkedList<Integer> select = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        dfsForPathSum(root, sum, select, res);
        return res;
    }

    private static void dfsForPathSum(TreeNode root, int sum, LinkedList<Integer> select, List<List<Integer>> res) {
        if (root == null) {
            return;
        }

        select.add(root.val);

        //因为左右字节点都为空时才可以判定结束，所以结束条件后置。
        if (root.val == sum && root.left == null && root.right == null) {
            res.add(new ArrayList<>(select));
            //不能有return,否则会少删除一个值。
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

        //左变右
        root.left = null;
        root.right = left;

        //老右粘到新右
        TreeNode p = root;
        while (p.right != null) {
            p = p.right;
        }
        p.right = right;
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

    //[129].求根到叶子节点的数字之和
    public static int sumNumbers(TreeNode root) {
        return count(root, 0);
    }

    private static int count(TreeNode root, int pre) {
        if (root == null) return 0;
        int value = pre * 10 + root.val;
        //当为叶子节点都为空的时候，需要返回自己的值
        if (root.left == null && root.right == null) return value;
        return count(root.left, value) + count(root.right, value);
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

    //[226].翻转二叉树
    public static TreeNode invertTree(TreeNode root) {
        if (root == null) return null;

        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;

        invertTree(root.left);
        invertTree(root.right);
        return root;
    }

    //[876]链表的中间结点
    public static ListNode middleNode(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
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

//        7. z字形变换
//        System.out.println(convert("LEETCODEISHIRING", 4));
//        System.out.println(convert("", 4));
//
//        28. 实现strStr()
//        System.out.println(strStr("abaclallb", "ll"));
//        System.out.println(strStr("aaaaa", "bba"));
//        System.out.println(strStr("hello", "ll"));
//        System.out.println(strStr("", "1"));
//        System.out.println(strStr("aaaaaab", "aab"));
//
//        49. 字母异位分组
//        System.out.println(groupAnagrams(new String[]{"eat", "tea", "tan", "ate", "nat", "bat"}));
//        System.out.println(groupAnagrams(new String[]{}));
//
//        单调栈
//        nextGreaterNumber(new int[]{1, 2, 3});
//        nextGreaterNumber(new int[]{2, 1, 2, 4, 3});
//
//        [94] 二叉树的中序遍历
//        TreeNode root = new TreeNode(1);
//        TreeNode r1 = new TreeNode(2);
//        TreeNode r2 = new TreeNode(3);
//        root.right = r1;
//        r1.left = r2;
//        System.out.println(inorderTraversal(root));
//        System.out.println(inorderTraversalV2(root));
//
//        [98].验证二叉搜索树
//        TreeNode root = new TreeNode(5);
//        TreeNode l1 = new TreeNode(1);
//        TreeNode r1 = new TreeNode(4);
//        TreeNode r2 = new TreeNode(3);
//        TreeNode r3 = new TreeNode(6);
//        root.left = l1;
//        root.right = r1;
//        r1.left = r2;
//        r1.right = r3;
//        System.out.println(isValidBST(root));
//        TreeNode root = new TreeNode(4);
//        TreeNode l1 = new TreeNode(1);
//        TreeNode l2 = new TreeNode(5);
//        TreeNode r1 = new TreeNode(6);
//        root.left = l1;
//        l1.right = l2;
//        root.right = r1;
//        System.out.println(isValidBST(root));
//
//        [100].相同的树
//        TreeNode one = new TreeNode(1);
//        TreeNode l1 = new TreeNode(2);
//        TreeNode l2 = new TreeNode(3);
//        TreeNode two = new TreeNode(1);
//        TreeNode r1 = new TreeNode(3);
//        TreeNode r2 = new TreeNode(2);
//        one.left = l1;
//        one.right = l2;
//        two.left = r1;
//        two.right = r2;
//        System.out.println(isSameTree(one, two));
//
//        [101].对称的二叉树
//        TreeNode one = new TreeNode(1);
//        TreeNode l1 = new TreeNode(2);
//        TreeNode r1 = new TreeNode(2);
//        TreeNode l2 = new TreeNode(3);
//        TreeNode l3 = new TreeNode(4);
//        TreeNode r2 = new TreeNode(4);
//        TreeNode r3 = new TreeNode(3);
//        one.left = l1;
//        one.right = r1;
//        l1.left = l2;
//        l1.right = l3;
//        r1.left = r2;
//        r1.right = r3;
//        System.out.println(isSymmetric(one));
//
//        102.二叉树的层序遍历
//        TreeNode root = new TreeNode(3);
//        TreeNode l1 = new TreeNode(9);
//        TreeNode r1 = new TreeNode(20);
//        TreeNode r2 = new TreeNode(15);
//        TreeNode r3 = new TreeNode(7);
//        root.left = l1;
//        root.right = r1;
//        r1.left = r2;
//        r1.right = r3;
//        System.out.println(levelOrder(root));
//
//        [103].二叉树的锯齿形层次遍历
//        TreeNode root = new TreeNode(3);
//        TreeNode l1 = new TreeNode(9);
//        TreeNode l2 = new TreeNode(8);
//        TreeNode l22 = new TreeNode(18);
//        TreeNode r1 = new TreeNode(20);
//        TreeNode r2 = new TreeNode(15);
//        TreeNode r21 = new TreeNode(6);
//        TreeNode r22 = new TreeNode(22);
//        TreeNode r3 = new TreeNode(7);
//        root.left = l1;
//        root.right = r1;
//        l1.left = l2;
//        l2.left = l22;
//        r1.left = r2;
//        r1.right = r3;
//        r2.left = r21;
//        r2.right = r22;
//        System.out.println(zigzagLevelOrder(root));
//
//        104.二叉树的最大深度
//        TreeNode root = new TreeNode(3);
//        TreeNode l1 = new TreeNode(9);
//        TreeNode r1 = new TreeNode(20);
//        TreeNode r2 = new TreeNode(15);
//        TreeNode r3 = new TreeNode(7);
//        root.left = l1;
//        root.right = r1;
//        r1.left = r2;
//        r1.right = r3;
//        System.out.println(maxDepth(root));
//
//        [105].从前序与中序遍历构造二叉树
//        TreeNode result = buildTree(new int[]{3, 9, 20, 15, 7}, new int[]{9, 3, 15, 20, 7});
//
//        [106].从中序与后序遍历构造二叉树
//        TreeNode result = buildTree2(new int[]{9, 3, 15, 20, 7}, new int[]{9, 15, 7, 20, 3});
//
//        [113].路径总和II
//        TreeNode root = new TreeNode(5);
//        TreeNode l1 = new TreeNode(4);
//        TreeNode r1 = new TreeNode(8);
//        TreeNode l2 = new TreeNode(11);
//        TreeNode l3 = new TreeNode(7);
//        TreeNode l4 = new TreeNode(2);
//        TreeNode r2 = new TreeNode(13);
//        TreeNode r3 = new TreeNode(4);
//        TreeNode r4 = new TreeNode(5);
//        TreeNode r5 = new TreeNode(1);
//        root.left = l1;
//        root.right = r1;
//        l1.left = l2;
//        l2.left = l3;
//        l2.right = l4;
//        r1.left = r2;
//        r1.right = r3;
//        r3.left = r4;
//        r3.right = r5;
//        System.out.println(pathSum(root, 22));
//
//        [114].二叉树展开为链表
//        TreeNode root = new TreeNode(1);
//        TreeNode l1 = new TreeNode(2);
//        TreeNode r1 = new TreeNode(5);
//        TreeNode l11 = new TreeNode(3);
//        TreeNode l12 = new TreeNode(4);
//        TreeNode r11 = new TreeNode(6);
//        root.left = l1;
//        root.right = r1;
//        r1.right = r11;
//        l1.left = l11;
//        l1.right = l12;
//        flatten(root);
//
        System.out.println(ladderLength("hit", "cog", Arrays.asList("hot", "dot", "dog", "lot", "log")));
        System.out.println(ladderLength("hit", "cog", Arrays.asList("hot", "dot", "dog", "lot", "log", "cog")));
//        [129].求根到叶子节点的数字之和
//        TreeNode root = new TreeNode(1);
//        TreeNode l1 = new TreeNode(2);
//        TreeNode r1 = new TreeNode(5);
//        TreeNode l11 = new TreeNode(3);
//        TreeNode l12 = new TreeNode(4);
//        TreeNode r11 = new TreeNode(6);
//        root.left = l1;
//        root.right = r1;
//        l1.left = l11;
//        l1.right = l12;
//        r1.right = r11;
//        System.out.println(sumNumbers(root));
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
//        [144].二叉树的前序遍历
//        TreeNode root = new TreeNode(1);
//        TreeNode r1 = new TreeNode(2);
//        TreeNode r2 = new TreeNode(3);
//        root.right = r1;
//        r1.left = r2;
//        System.out.println(preorderTraversalV2(root));
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

//        System.out.println(compareVersion("0.1", "1.1"));
//        System.out.println(compareVersion("1.0.1", "1"));
//        System.out.println(compareVersion("1.01", "1.001"));

//        System.out.println(fractionToDecimal(-4, 17));
//        System.out.println(fractionToDecimal(2, 3));
//
//        Node f1 = new Node(1);
//        Node f2 = new Node(2);
//        Node f3 = new Node(3);
//        Node f4 = new Node(4);
//        f1.neighbors = Arrays.asList(f2, f4);
//        f2.neighbors = Arrays.asList(f1, f3);
//        f3.neighbors = Arrays.asList(f2, f4);
//        f4.neighbors = Arrays.asList(f1, f3);
//
//        Node res = cloneGraph(f1);
//        [226].翻转二叉树
//        TreeNode root = new TreeNode(1);
//        TreeNode l1 = new TreeNode(2);
//        TreeNode r1 = new TreeNode(5);
//        TreeNode l11 = new TreeNode(3);
//        TreeNode l12 = new TreeNode(4);
//        TreeNode r11 = new TreeNode(6);
//        root.left = l1;
//        root.right = r1;
//        l1.left = l11;
//        l1.right = l12;
//        r1.right = r11;
//        TreeNode result = invertTree(root);
    }
}
