package com.owen.algorithm;

import com.owen.algorithm.LinkList.ListNode;

import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public class Tree {

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
        public List<Node> children;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, List<Node> _children) {
            val = _val;
            children = _children;
        }
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

    //[95].不同的二叉搜索树II
    public static List<TreeNode> generateTrees(int n) {
        if (n == 0) return new LinkedList<>();
        return build(1, n);
    }

    private static List<TreeNode> build(int start, int end) {
        List<TreeNode> res = new LinkedList<>();
        if (start > end) {
            res.add(null);
            return res;
        }

        //选择根节点
        for (int i = start; i <= end; i++) {
            List<TreeNode> leftTree = build(start, i - 1);
            List<TreeNode> rightTree = build(i + 1, end);
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
        //根 左 右
        //左 根 右
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

    //[297].二叉树的序列化与反序列化
    public static class Codec {
        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            if (root == null) return "#";
            return root.val + "," + serialize(root.left) + "," + serialize(root.right);
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            if (data.equals("#")) return null;
            LinkedList<String> queue = new LinkedList(Arrays.asList(data.split(",")));
            return myDes(queue);
        }

        private TreeNode myDes(LinkedList<String> queue) {
            String top = queue.pop();
            if (top.equals("#")) {
                return null;
            }
            TreeNode root = new TreeNode(Integer.parseInt(top));
            root.left = myDes(queue);
            root.right = myDes(queue);
            return root;
        }
    }

    //[421].数组中两个数的最大异或值
    public static class Solution421 {
        class TrieNode {
            TrieNode[] son = new TrieNode[2];
        }

        TrieNode root = new TrieNode();

        private void build(int[] nums) {
            for (int num : nums) {
                TrieNode cur = root;
                for (int i = 30; i >= 0; i--) {
                    int bit = num >> i & 1;
                    if (cur.son[bit] == null) {
                        cur.son[bit] = new TrieNode();
                    }
                    cur = cur.son[bit];
                }
            }
        }

        private int search(int num) {
            TrieNode cur = root;
            int res = 0;
            for (int i = 30; i >= 0; i--) {
                int bit = num >> i & 1;
                //取相反的路径
                if (cur.son[bit ^ 1] != null) {
                    res += 1 << i;
                    cur = cur.son[bit ^ 1];
                } else {
                    cur = cur.son[bit];
                }
            }
            return res;
        }

        public int findMaximumXOR(int[] nums) {
            if (nums.length == 1) return nums[0];
            build(nums);
            int max = 0;
            for (int num : nums) {
                max = Math.max(max, search(num));
            }
            return max;
        }
    }

    //[427].建立四叉树
    public static class Solution427 {
        class Node {
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
            return helper(grid, 0, 0, grid.length);
        }

        private Node helper(int[][] grid, int i, int j, int len) {
            if (len == 1) {
                return new Node(grid[i][j] == 1, true);
            }

            int k = len / 2;
            Node tl = helper(grid, i, j, k);
            Node tr = helper(grid, i, j + k, k);
            Node bl = helper(grid, i + k, j, k);
            Node br = helper(grid, i + k, j + k, k);
            if (tl.val == tr.val && tr.val == bl.val && bl.val == br.val && tl.isLeaf && tr.isLeaf && bl.isLeaf && br.isLeaf) {
                return new Node(tl.val, true);
            } else {
                return new Node(true, false, tl, tr, bl, br);
            }
        }
    }

    //[429]. N叉树的层序遍历
    public List<List<Integer>> levelOrder(Node root) {
        if (root == null) return new ArrayList<>();
        LinkedList<Node> queue = new LinkedList<>();
        queue.add(root);

        List<List<Integer>> res = new ArrayList<>();
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> arr = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                Node cur = queue.poll();
                arr.add(cur.val);
                if (!cur.children.isEmpty()) {
                    for (Node child : cur.children) {
                        queue.add(child);
                    }
                }
            }
            res.add(arr);
        }
        return res;
    }

    //[437].路径总和III
    public static int pathSum3(TreeNode root, int sum) {
        if (root == null) return 0;
        int result = countPath(root, sum);
        int left = pathSum3(root.left, sum);
        int right = pathSum3(root.right, sum);
        return result + left + right;
    }

    private static int countPath(TreeNode root, int sum) {
        if (root == null) return 0;

        sum = sum - root.val;
        int rootNum = sum == 0 ? 1 : 0;
        int leftNum = countPath(root.left, sum);
        int rightNum = countPath(root.right, sum);

        return rootNum + leftNum + rightNum;
    }

    //[449].序列化和反序列化二叉搜索树
    public static class Codec2 {

        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            if (root == null) return "";
            StringBuilder res = new StringBuilder();
            helper(root, res);
            return res.substring(0, res.length() - 1);
        }

        private void helper(TreeNode root, StringBuilder sb) {
            if (root == null) {
                return;
            }
            sb.append(root.val).append(',');
            helper(root.left, sb);
            helper(root.right, sb);
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            if (data == null || data.length() == 0) return null;
            String[] arr = data.split(",");
            return helper(arr, 0, arr.length - 1);
        }

        private TreeNode helper(String[] data, int low, int high) {
            if (low > high) return null;
            //根左右
            //4,3,2,5
            TreeNode root = new TreeNode(Integer.parseInt(data[low]));
            //找到比当前根节点大的节点，就是左右分界点
            int index = high + 1;
            for (int i = low + 1; i <= high; i++) {
                if (Integer.parseInt(data[i]) > root.val) {
                    index = i;
                    break;
                }
            }

            root.left = helper(data, low + 1, index - 1);
            root.right = helper(data, index, high);
            return root;
        }
    }

    //[450].删除二叉搜索树中的节点
    public static TreeNode deleteNode(TreeNode root, int key) {
        if (root == null) return null;
        if (root.val == key) {
            if (root.left == null) return root.right;
            if (root.right == null) return root.left;
            if (root.left != null && root.right != null) {
                TreeNode minNode = getMin(root.right);
                root.val = minNode.val;
                root.right = deleteNode(root.right, minNode.val);
            }
        } else if (root.val < key) {
            root.right = deleteNode(root.right, key);
        } else {
            root.left = deleteNode(root.left, key);
        }
        return root;
    }

    private static TreeNode getMin(TreeNode root) {
        while (root.left != null) root = root.left;
        return root;
    }

    //[508].出现次数最多的子树元素和
    public static int[] findFrequentTreeSum(TreeNode root) {
        Map<Integer, Integer> countMap = new HashMap<>();
        AtomicInteger maxCount = new AtomicInteger();
        dfsForTreeSum(root, countMap, maxCount);

        List<Integer> res = new ArrayList<>();
        for (int num : countMap.keySet()) {
            if (countMap.get(num).equals(maxCount.get())) {
                res.add(num);
            }
        }
        int[] result = new int[res.size()];
        for (int i = 0; i < res.size(); i++) {
            result[i] = res.get(i);
        }
        return result;
    }

    private static int dfsForTreeSum(TreeNode root, Map<Integer, Integer> res, AtomicInteger max) {
        if (root == null) return 0;

        int sum = root.val + dfsForTreeSum(root.left, res, max) + dfsForTreeSum(root.right, res, max);
        res.put(sum, res.getOrDefault(sum, 0) + 1);

        max.set(Math.max(max.get(), res.get(sum)));
        return sum;
    }

    //[513].找树左下角的值
    public static int findBottomLeftValue(TreeNode root) {
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int res = root.val;
        while (!queue.isEmpty()) {
            res = queue.peek().val;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();

                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }
            }
        }
        return res;
    }

    //[515].在每个树行中找最大值
    public static List<Integer> largestValues(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            int max = Integer.MIN_VALUE;
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();
                max = Math.max(max, cur.val);

                if (cur.left != null) {
                    queue.offer(cur.left);
                }

                if (cur.right != null) {
                    queue.offer(cur.right);
                }
            }
            res.add(max);
        }
        return res;
    }

    //[538].把二叉搜索树转换为累加树
    public static TreeNode convertBST(TreeNode root) {
        traverseBST(root, new AtomicInteger(0));
        return root;
    }

    private static void traverseBST(TreeNode root, AtomicInteger sum) {
        if (root == null)
            return;
        traverseBST(root.right, sum);

        sum.addAndGet(root.val);
        root.val = sum.get();

        traverseBST(root.left, sum);
    }

    public static int maxDepth(Node root) {
        if (root == null) return 0;
        int childMaxDepth = 0;

        if (root.children != null) {
            for (Node child : root.children) {
                childMaxDepth = Math.max(maxDepth(child), childMaxDepth);
            }
        }
        return childMaxDepth + 1;
    }

    //[589].N叉树的前序遍历
    public static List<Integer> preorder(Node root) {
        List<Integer> res = new ArrayList<>();
        preorderN(root, res);
        return res;
    }

    private static void preorderN(Node root, List<Integer> res) {
        if (root == null) {
            return;
        }
        res.add(root.val);

        if (root.children != null) {
            for (Node child : root.children) {
                preorderN(child, res);
            }
        }
    }

    //[590].N叉树的后序遍历
    public List<Integer> postorder(Node root) {
        List<Integer> res = new ArrayList<>();
        postorderN(root, res);
        return res;
    }

    private static void postorderN(Node root, List<Integer> res) {
        if (root == null) return;

        if (root.children != null) {
            for (Node child : root.children) {
                postorderN(child, res);
            }
        }

        res.add(root.val);
    }

    //[623].在二叉树中增加一行
    public static TreeNode addOneRow(TreeNode root, int v, int d) {
        if (d == 1) {
            TreeNode newOne = new TreeNode(v);
            newOne.left = root;
            return newOne;
        }
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int layer = 1;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();
                if (layer == d - 1) {
                    TreeNode newLeft = new TreeNode(v);
                    TreeNode newRight = new TreeNode(v);
                    newLeft.left = cur.left;
                    newRight.right = cur.right;

                    cur.left = newLeft;
                    cur.right = newRight;
                } else {
                    if (cur.left != null) {
                        queue.offer(cur.left);
                    }
                    if (cur.right != null) {
                        queue.offer(cur.right);
                    }
                }
            }
            layer++;
        }
        return root;
    }

    //[700].二叉搜索树中的搜索
    public static TreeNode searchBST(TreeNode root, int val) {
        if (root == null) {
            return null;
        }
        if (root.val == val) {
            return root;
        } else if (root.val < val) {
            return searchBST(root.right, val);
        } else {
            return searchBST(root.left, val);
        }
    }

    //[701].二叉搜索树中插入操作
    public static TreeNode insertIntoBST(TreeNode root, int val) {
        if (root == null) return new TreeNode(val);

        if (root.val < val) {
            root.right = insertIntoBST(root.right, val);
        }
        if (root.val > val) {
            root.left = insertIntoBST(root.left, val);
        }
        return root;
    }

    //[1373].二叉搜索子树的最大健和值
    public static int maxSumBST(TreeNode root) {
        AtomicInteger res = new AtomicInteger();
        traverse(root, res);
        return res.get();
    }

    private static int[] traverse(TreeNode root, AtomicInteger maxSum) {
        //是否为BST 最小值 最大值 健和值
        if (root == null) {
            return new int[]{1, Integer.MAX_VALUE, Integer.MIN_VALUE, 0};
        }

        int[] left = traverse(root.left, maxSum);
        int[] right = traverse(root.right, maxSum);

        int[] res = new int[4];
        //可能是BST树， 根节点 > 左子树的最大值 且 根节点 < 右子树的最小值
        if (left[0] == 1 && right[0] == 1
                && left[2] < root.val
                && root.val < right[1]) {

            res[0] = 1;
            // 左子树的最小值 根节点 取最小
            res[1] = Math.min(root.val, left[1]);
            // 右子树的最大值 根节点 取最大
            res[2] = Math.max(root.val, right[2]);

            res[3] = left[3] + right[3] + root.val;
            maxSum.set(Math.max(maxSum.get(), res[3]));
        } else {
            res[0] = 0;
        }
        return res;
    }

    public static void main(String[] args) {
//        [94]. 二叉树的中序遍历
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
//        one.left  = new TreeNode(2);
//        one.left.left = new TreeNode(3);
//        one.left.right = new TreeNode(4);
//        one.right = new TreeNode(2);
//        one.right.left = new TreeNode(4);
//        one.right.right = new TreeNode(3);
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
//        [144].二叉树的前序遍历
//        TreeNode root = new TreeNode(1);
//        TreeNode r1 = new TreeNode(2);
//        TreeNode r2 = new TreeNode(3);
//        root.right = r1;
//        r1.left = r2;
//        System.out.println(preorderTraversalV2(root));
//
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
//        [297].二叉树的序列化和反序列化
//        Codec code =  new Codec();
//        TreeNode root = new TreeNode(1);
//        root.left = new TreeNode(2);
//        root.right = new TreeNode(3);
//        root.right.left = new TreeNode(4);
//        String a = code.serialize(root);
//        System.out.println(a);
//        TreeNode res = code.deserialize(a);
//
//        [421].数组中两个数的最大异或值
//        System.out.println(new Solution421().findMaximumXOR(new int[]{3, 10, 5, 25, 2, 8}));
//
//        [427].建立四叉树
//        Solution427.Node node  = new Solution427().construct(new int[][]{{1, 1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 0, 0, 0, 0}});
//        System.out.println();
//
//        [449].序列化和反序列化二叉搜索树
//        Codec2 codec = new Codec2();
//        TreeNode root = new TreeNode(3);
//        root.left = new TreeNode(1);
//        root.left.right = new TreeNode(2);
//        root.right = new TreeNode(4);
//        String serialize = codec.serialize(root);
//        System.out.println(serialize);
//        TreeNode res = codec.deserialize(serialize);
//        System.out.println();
//
//        [508].出现次数最多的子树元素和
//        TreeNode root = new TreeNode(5);
//        root.left = new TreeNode(2);
//        root.right = new TreeNode(-5);
//        System.out.println(Arrays.toString(findFrequentTreeSum(root)));;
//        root.left = new TreeNode(2);
//        root.right = new TreeNode(-3);
//        System.out.println(Arrays.toString(findFrequentTreeSum(root)));;
//
//        [513].找树左下角的值
//        TreeNode root = new TreeNode(1);
//        root.left = new TreeNode(2);
//        root.left.left = new TreeNode(4);
//        root.right = new TreeNode(3, new TreeNode(5), new TreeNode(6));
//        root.right.left.left = new TreeNode(7);
//        System.out.println(findBottomLeftValue(root));
//
//        [515].在每个树行中找最大值
//        TreeNode root = new TreeNode(1);
//        root.left = new TreeNode(3);
//        root.right = new TreeNode(2);
//        root.left.left = new TreeNode(5);
//        root.left.right = new TreeNode(3);
//        root.right.right = new TreeNode(9);
//        System.out.println(largestValues(root));
//
//        [623].在二叉树中增加一行
//        TreeNode root = new TreeNode(4);
//        root.left = new TreeNode(2);
//        root.right = new TreeNode(6);
//        root.left.left = new TreeNode(3);
//        root.left.right = new TreeNode(1);
//        root.right.left = new TreeNode(5);
//        TreeNode newOne = addOneRow(root, 1,2);
    }
}
