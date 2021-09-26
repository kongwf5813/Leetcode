package com.owen.algorithm.homework;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Random;
import java.util.Stack;
import java.util.concurrent.LinkedBlockingQueue;

public class Homework {
    public static class ListNode {
        public int val;
        public ListNode next;

        public ListNode(int x) {
            val = x;
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

    //6x-1作业：用队列实现栈
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

    //6x-1作业：用栈实现队列
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

    //7-7作业：层序树的遍历
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

    //8-1作业：链表的递归实现，Leetcode 203，移除链表元素
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

    //8-7作业：链表的递归实现，Leetcode 25，有效括号
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

    //8x-1作业：Leetcode 206, 反转链表 迭代实现
    public static ListNode reverse(ListNode start, ListNode end) {
        //注意一定要三个节点，否则可能死循环
        ListNode pre = null, cur = start, next;
        while (cur != end) {
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        //返回的是遍历到最后的节点
        return pre;
    }

    //8x-1作业：Leetcode 206, 反转链表 递归实现
    public static ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode last = reverseList(head.next);
        head.next.next = head;
        //断链呀
        head.next = null;
        return last;
    }

    //9-4作业：归并排序
    public static class MergeSort {

        private MergeSort() {
        }

        public static <E extends Comparable<E>> void sort(E[] arr) {
            sort(arr, 0, arr.length - 1);
        }

        private static <E extends Comparable<E>> void sort(E[] arr, int left, int right) {
            if (left >= right) return;
            int mid = left + (right - left) / 2;
            sort(arr, left, mid);
            sort(arr, mid + 1, right);
            //后序遍历
            merge(arr, left, mid, right);
        }

        private static <E extends Comparable<E>> void merge(E[] arr, int left, int mid, int right) {
            E[] temp = Arrays.copyOfRange(arr, left, right + 1);
            int i = left, j = mid + 1;
            for (int k = left; k <= right; k++) {
                if (i > mid) {
                    arr[k] = temp[(j++) - left];
                } else if (j > right) {
                    arr[k] = temp[(i++) - left];
                } else if (temp[i - left].compareTo(temp[j - left]) <= 0) {
                    arr[k] = temp[(i++) - left];
                } else {
                    arr[k] = temp[(j++) - left];
                }
            }
        }
    }

    //11-9作业：只创建一个 Random 类
    public static class ArrayGenerator {
        public static Integer[] generateRandomByBound(int n, int bound) {
            Integer[] arr = new Integer[n];
            Random random = new Random();
            for (int i = 0; i < n; i++)
                arr[i] = random.nextInt(bound);
            return arr;
        }
    }

    //11-9作业：只创建一个 Random 类， 抽样蓄水问题
    static class Solution {
        private int[] original;

        public Solution(int[] nums) {
            original = nums;
        }

        /**
         * Resets the array to its original configuration and return it.
         */
        public int[] reset() {
            return original;
        }

        /**
         * Returns a random shuffling of the array.
         */
        public int[] shuffle() {
            Random random = new Random();
            int[] arr = original.clone();
            for (int i = 0; i < arr.length; i++) {
                int randomIdx = random.nextInt(arr.length - i) + i;
                int temp = arr[i];
                arr[i] = arr[randomIdx];
                arr[randomIdx] = temp;
            }
            return arr;
        }
    }

    //12-8作业: Leetcode 75 Sort Colors
    public static void sortColors(int[] nums) {
        int cur = 0, start = 0, end = nums.length - 1;
        while (cur <= end) {
            if (nums[cur] == 2) {
                int temp = nums[end];
                nums[end] = nums[cur];
                nums[cur] = temp;
                end--;
            } else if (nums[cur] == 1) {
                cur++;
            } else if (nums[cur] == 0) {
                int temp = nums[start];
                nums[start] = nums[cur];
                nums[cur] = temp;
                start++;
                cur++;
            }
        }
    }

    //12-10作业： Select K 问题
    public static int findKthLargest(int[] nums, int k) {
        //第K大意味着是从小到大是第n-k位
        return quickSort(nums, 0, nums.length - 1, nums.length - k);
    }

    private static int quickSort(int[] nums, int left, int right, int index) {
        int l = left, r = right;
        int pivot = nums[l];
        while (l < r) {
            while (l < r && pivot <= nums[r]) r--;
            //右边的小值赋值给左边
            nums[l] = nums[r];

            while (l < r && nums[l] <= pivot) l++;
            //左边的大值赋值给右边
            nums[r] = nums[l];
        }
        nums[l] = pivot;

        if (l == index) {
            return nums[l];
        } else if (l > index) {
            return quickSort(nums, left, l - 1, index);
        } else {
            return quickSort(nums, l + 1, right, index);
        }
    }

    //14x-3作业：Leetcode 1011 在D天内送达包裹的能力
    public static int shipWithinDays(int[] weights, int D) {
        int left = Arrays.stream(weights).max().getAsInt(), right = Arrays.stream(weights).sum();
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (needDays(weights, mid) <= D) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    private static int needDays(int[] weights, int cap) {
        int days = 0;
        int temp = 0;
        for (int weight : weights) {
            temp += weight;
            if (temp > cap) {
                days++;
                temp = weight;
            }
        }
        if (temp <= cap) {
            days++;
        }
        return days;
    }

    //15-9作业： 二叉树的中序遍历(迭代)
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

    //15-9作业： 二叉树的后序遍历(迭代)
    public static List<Integer> postorderTraversal(TreeNode root) {
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

    //23-4作业：Trie字典树的前缀查询
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

    //24-6作业：并查集 压缩
    public static class UnionFind {
        private int[] parent;
        private int[] size;

        public UnionFind(int count) {
            parent = new int[count];
            size = new int[count];
            for (int i = 0; i < count; i++) {
                size[i] = 1;
                parent[i] = i;
            }
        }

        public void union(int p, int q) {
            int rootP = find(p);
            int rootQ = find(q);

            if (size[rootP] > size[rootQ]) {
                parent[rootQ] = rootP;
                size[rootP] += size[rootQ];
            } else {
                parent[rootP] = rootQ;
                size[rootQ] += size[rootP];
            }
        }

        public int find(int x) {
            while (x != parent[x]) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }

        public int getParent(int i) {
            return parent[i];
        }

        public boolean connected(int p, int q) {
            int rootP = find(p);
            int rootQ = find(q);
            return rootP == rootQ;
        }
    }

    //26-8作业：红黑树
    public static class RBTree<K extends Comparable<K>, V> {

        private static final boolean RED = true;
        private static final boolean BLACK = false;

        private class Node{
            public K key;
            public V value;
            public Node left, right;
            public boolean color;

            public Node(K key, V value){
                this.key = key;
                this.value = value;
                left = null;
                right = null;
                color = RED;
            }
        }

        private Node root;
        private int size;

        public RBTree(){
            root = null;
            size = 0;
        }

        public int getSize(){
            return size;
        }

        public boolean isEmpty(){
            return size == 0;
        }

        // 判断节点node的颜色
        private boolean isRed(Node node){
            if(node == null)
                return BLACK;
            return node.color;
        }

        //   node                     x
        //  /   \     左旋转         /  \
        // T1   x   --------->   node   T3
        //     / \              /   \
        //    T2 T3            T1   T2
        private Node leftRotate(Node node){

            Node x = node.right;

            // 左旋转
            node.right = x.left;
            x.left = node;

            x.color = node.color;
            node.color = RED;

            return x;
        }

        //     node                   x
        //    /   \     右旋转       /  \
        //   x    T2   ------->   y   node
        //  / \                       /  \
        // y  T1                     T1  T2
        private Node rightRotate(Node node){

            Node x = node.left;

            // 右旋转
            node.left = x.right;
            x.right = node;

            x.color = node.color;
            node.color = RED;

            return x;
        }

        // 颜色翻转
        private void flipColors(Node node){
            node.color = RED;
            node.left.color = BLACK;
            node.right.color = BLACK;
        }

        // 向红黑树中添加新的元素(key, value)
        public void add(K key, V value){
            root = add(root, key, value);
            root.color = BLACK; // 最终根节点为黑色节点
        }

        // 向以node为根的红黑树中插入元素(key, value)，递归算法
        // 返回插入新节点后红黑树的根
        private Node add(Node node, K key, V value){

            if(node == null){
                size ++;
                return new Node(key, value); // 默认插入红色节点
            }

            if(key.compareTo(node.key) < 0)
                node.left = add(node.left, key, value);
            else if(key.compareTo(node.key) > 0)
                node.right = add(node.right, key, value);
            else // key.compareTo(node.key) == 0
                node.value = value;

            if (isRed(node.right) && !isRed(node.left))
                node = leftRotate(node);

            if (isRed(node.left) && isRed(node.left.left))
                node = rightRotate(node);

            if (isRed(node.left) && isRed(node.right))
                flipColors(node);

            return node;
        }

        // 返回以node为根节点的二分搜索树中，key所在的节点
        private Node getNode(Node node, K key){

            if(node == null)
                return null;

            if(key.equals(node.key))
                return node;
            else if(key.compareTo(node.key) < 0)
                return getNode(node.left, key);
            else // if(key.compareTo(node.key) > 0)
                return getNode(node.right, key);
        }

        public boolean contains(K key){
            return getNode(root, key) != null;
        }

        public V get(K key){

            Node node = getNode(root, key);
            return node == null ? null : node.value;
        }

        public void set(K key, V newValue){
            Node node = getNode(root, key);
            if(node == null)
                throw new IllegalArgumentException(key + " doesn't exist!");

            node.value = newValue;
        }

        // 返回以node为根的二分搜索树的最小值所在的节点
        private Node minimum(Node node){
            if(node.left == null)
                return node;
            return minimum(node.left);
        }

        // 删除掉以node为根的二分搜索树中的最小节点
        // 返回删除节点后新的二分搜索树的根
        private Node removeMin(Node node){

            if(node.left == null){
                Node rightNode = node.right;
                node.right = null;
                size --;
                return rightNode;
            }

            node.left = removeMin(node.left);
            return node;
        }

        // 从二分搜索树中删除键为key的节点
        public V remove(K key){

            Node node = getNode(root, key);
            if(node != null){
                root = remove(root, key);
                return node.value;
            }
            return null;
        }

        private Node remove(Node node, K key){

            if( node == null )
                return null;

            if( key.compareTo(node.key) < 0 ){
                node.left = remove(node.left , key);
                return node;
            }
            else if(key.compareTo(node.key) > 0 ){
                node.right = remove(node.right, key);
                return node;
            }
            else{   // key.compareTo(node.key) == 0

                // 待删除节点左子树为空的情况
                if(node.left == null){
                    Node rightNode = node.right;
                    node.right = null;
                    size --;
                    return rightNode;
                }

                // 待删除节点右子树为空的情况
                if(node.right == null){
                    Node leftNode = node.left;
                    node.left = null;
                    size --;
                    return leftNode;
                }

                // 待删除节点左右子树均不为空的情况

                // 找到比待删除节点大的最小节点, 即待删除节点右子树的最小节点
                // 用这个节点顶替待删除节点的位置
                Node successor = minimum(node.right);
                successor.right = removeMin(node.right);
                successor.left = node.left;
                node.left = node.right = null;
                return successor;
            }
        }
    }
}
