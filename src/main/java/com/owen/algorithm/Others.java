package com.owen.algorithm;

import java.util.*;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.zip.CheckedOutputStream;

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

    //[173].二叉搜索树迭代器
    class BSTIterator {
        Stack<TreeNode> stack;

        public BSTIterator(TreeNode root) {
            stack = new Stack<>();
            pushMin(root);
        }

        /**
         * @return the next smallest number
         */
        public int next() {
            TreeNode top = stack.pop();
            pushMin(top.right);
            return top.val;
        }

        /**
         * @return whether we have a next smallest number
         */
        public boolean hasNext() {
            return !stack.isEmpty();
        }

        public void pushMin(TreeNode root) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
        }
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

    //[199]二叉树的右视图
    public static List<Integer> rightSideView(TreeNode root) {
        Queue<TreeNode> queue = new ArrayDeque<>();
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        queue.add(root);

        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode top = queue.poll();
                //每一层的最后一个
                if (i == size - 1) {
                    res.add(top.val);
                }

                if (top.left != null) {
                    queue.add(top.left);
                }
                if (top.right != null) {
                    queue.add(top.right);
                }
            }
        }
        return res;
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

    //[208].实现Trie(前缀树)
    public static class Trie {
        boolean isEnd;
        Trie[] next;

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
            char[] chars = word.toCharArray();
            for (char single : chars) {
                int index = single - 'a';
                Trie next = root.next[index];
                if (next == null) {
                    next = new Trie();
                    root.next[index] = next;
                }
                root = next;
            }
            root.isEnd = true;
        }

        /**
         * Returns if the word is in the trie.
         */
        public boolean search(String word) {
            char[] chars = word.toCharArray();
            Trie root = this;
            for (char single : chars) {
                int index = single - 'a';
                Trie next = root.next[index];
                if (next == null) {
                    return false;
                }
                root = next;
            }
            return root.isEnd;
        }

        /**
         * Returns if there is any word in the trie that starts with the given prefix.
         */
        public boolean startsWith(String prefix) {
            char[] chars = prefix.toCharArray();
            Trie root = this;
            for (char single : chars) {
                int index = single - 'a';
                Trie next = root.next[index];
                if (next == null) {
                    return false;
                }
                root = next;
            }
            return true;
        }
    }

    //[222].完全二叉树的节点个数
    public static int countNodes(TreeNode root) {
        if (root == null) return 0;

        int leftLevel = countLevel(root.left);
        int rightLevel = countLevel(root.right);

        if (leftLevel == rightLevel) {
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
            //最左边的节点就是层高
            root = root.left;
        }
        return level;
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

    //[230].二叉搜索数中第K小的元素
    public static int kthSmallest(TreeNode root, int k) {
        AtomicInteger res = new AtomicInteger();
        dfsForKthSmallest(root, k, new AtomicInteger(0), res);
        return res.get();
    }

    private static void dfsForKthSmallest(TreeNode root, int k, AtomicInteger count, AtomicInteger res) {
        if (root == null) {
            return;
        }

        dfsForKthSmallest(root.left, k, count, res);
        if (count.incrementAndGet() == k) {
            res.addAndGet(root.val);
            return;
        }
        dfsForKthSmallest(root.right, k, count, res);
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

    //[235].二叉搜索数的最近公共祖先
    public static TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root.val > p.val && root.val > q.val) return lowestCommonAncestor(root.left, p, q);
        else if (root.val < p.val && root.val < q.val)
            return lowestCommonAncestor(root.right, p, q);
        else return root;
    }

    //[236].二叉树的最近公共祖先
    public static TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor2(root.left, p, q);
        TreeNode right = lowestCommonAncestor2(root.right, p, q);
        if (left == null) return right;
        if (right == null) return left;
        return root;
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

    //[257].二叉树的所有路径
    public static List<String> binaryTreePaths(TreeNode root) {
        List<String> res = new ArrayList<>();
        dfsForBinaryTreePaths(res, "", root);
        return res;
    }

    private static void dfsForBinaryTreePaths(List<String> res, String path, TreeNode root) {
        if (root == null) {
            return;
        }

        path += root.val + "->";
        //左右字节点都为空才是叶子节点
        if (root.left == null && root.right == null) {
            res.add(path.substring(0, path.length() - 2));
            return;
        }
        dfsForBinaryTreePaths(res, path, root.left);
        dfsForBinaryTreePaths(res, path, root.right);
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
        if (size == 0 ) return 0;
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

    //[876].链表的中间结点
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
//        [127].单词接龙
//        System.out.println(ladderLength("hit", "cog", Arrays.asList("hot", "dot", "dog", "lot", "log")));
//        System.out.println(ladderLength("hit", "cog", Arrays.asList("hot", "dot", "dog", "lot", "log", "cog")));
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

//        [199].二叉树的右视图
//        TreeNode root = new TreeNode(1);
//        TreeNode l1 = new TreeNode(2);
//        TreeNode r1 = new TreeNode(3);
//        TreeNode l2 = new TreeNode(5);
//        TreeNode r2 = new TreeNode(4);
//        root.left = l1;
//        root.right = r1;
//        l1.right = l2;
//        r1.left = r2;
//        System.out.println(rightSideView(root));
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
//        [208].实现Trie(前缀树)
//        Trie trie = new Trie();
//        trie.insert("apple");
//        System.out.println(trie.search("apple"));   // 返回 true
//        System.out.println(trie.search("app"));     // 返回 false
//        System.out.println(trie.startsWith("app")); // 返回 true
//        trie.insert("app");
//        System.out.println(trie.search("app"));     // 返回 true
//        trie.insert("banana");
//        System.out.println(trie.startsWith("banana"));
//
//        [222].完全二叉树的节点个数
//        TreeNode root = new TreeNode(1);
//        TreeNode l1 = new TreeNode(2);
//        TreeNode r1 = new TreeNode(3);
//        TreeNode l11 = new TreeNode(4);
//        TreeNode l12 = new TreeNode(5);
//        root.left = l1;
//        root.right = r1;
//        l1.left = l11;
//        l1.right = l12;
//        System.out.println(countNodes(root));
//
//        [223].矩形面积
//        System.out.println(computeArea(-3, 0, 3, 4, 0, -1, 9, 2));
//
//        [224].基本计算器
//        System.out.println(calculate("(1+(4+5+2)-3)"));
//        System.out.println(calculate("(1+(4+5+2)-3)+(6+8)"));
//        System.out.println(calculate("(1+2"));
//
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
//
//        [227]基本计算器II
//        System.out.println(calculate2("3+2*2"));
//        System.out.println(calculate2(" 3+5 / 2 "));
//        System.out.println(calculate2(" 3/2 "));
//        System.out.println(calculate2(" 1 "));
//
//        [230].二叉搜索数中第K小的元素
//        TreeNode root = new TreeNode(3);
//        TreeNode l1 = new TreeNode(1);
//        TreeNode l2 = new TreeNode(2);
//        TreeNode r1 = new TreeNode(4);
//        root.left = l1;
//        l1.right = l2;
//        root.right = r1;
//        System.out.println(kthSmallest(root, 5));
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

//        [257].二叉树的所有路径
//        TreeNode root = new TreeNode(1);
//        TreeNode l1 = new TreeNode(2);
//        TreeNode l2 = new TreeNode(5);
//        TreeNode r1 = new TreeNode(3);
//        root.left = l1;
//        l1.right = l2;
//        root.right = r1;
//        System.out.println(binaryTreePaths(root));
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
        System.out.println(maxProduct(new String[]{"abcw", "baz", "foo", "bar", "xtfn", "abcdef"}));
        System.out.println(maxProduct(new String[]{"a", "ab", "abc", "d", "cd", "bcd", "abcd"}));
        System.out.println(maxProduct(new String[]{"a", "aa", "aaa", "aaaa"}));

        //[319].灯泡开关
//        System.out.println(bulbSwitch(12));
    }
}
