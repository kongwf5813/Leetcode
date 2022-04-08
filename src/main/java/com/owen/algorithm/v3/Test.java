package com.owen.algorithm.v3;


import com.owen.algorithm.v3.AllOfThem.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class Test {


    public static void main(String[] args) {
//        int[] arr = new int[]{2, 6, 4, 3, 2, 1};
//        new Test().heapSort(arr);
//        System.out.println(Arrays.toString(arr));
//        TreeNode root = new TreeNode(8);
//        root.left = new TreeNode(9);
//        root.right = new TreeNode(20);
//        root.right.left = new TreeNode(15);
//        root.right.right = new TreeNode(7);
//        new Test().maxPathSum(root);
        new Test().largestRectangleArea(new int[]{2, 1, 5, 6, 2, 3});
    }

    void heapSort(int[] nums) {
        int size = nums.length;
        buildMaxHeap(nums, size);
        for (int i = size - 1; i > 0; i--) {
            swap(nums, 0, i);
            maxHeapify(nums, 0, i);
        }
    }


    void buildMaxHeap(int[] nums, int size) {
        for (int i = size / 2; i >= 0; i--) {
            maxHeapify(nums, i, size);
        }
    }

    void maxHeapify(int[] nums, int i, int size) {
        int left = 2 * i + 1, right = 2 * i + 2;
        int largest = i;
        if (left < size && nums[left] > nums[largest]) largest = left;
        if (right < size && nums[right] > nums[largest]) largest = right;
        if (largest != i) {
            swap(nums, i, largest);
            maxHeapify(nums, largest, size);
        }
    }

    void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }


    public int largestRectangleArea(int[] heights) {
        int[] newHeights = new int[heights.length + 2];
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


    int ans = Integer.MIN_VALUE;
    List<Integer> path = new ArrayList<>();

    public int maxPathSum(TreeNode root) {
        dfs(root);
        System.out.println(path);
        return ans;
    }

    private Pair dfs(TreeNode root) {
        if (root == null) {
            return new Pair(0, new ArrayList<>());
        }
        Pair left = dfs(root.left);
        Pair right = dfs(root.right);

        List<Integer> path = new ArrayList<>();
        int res = root.val;
        if (left.sum > 0 && left.sum > right.sum) {
            path.addAll(left.path);
            path.add(root.val);
            res += left.sum;
        } else if (right.sum > 0 && right.sum > left.sum) {
            path.addAll(right.path);
            path.add(root.val);
            res += right.sum;
        } else {
            path.add(root.val);
        }

        if (res > ans) {
            this.ans = res;
            this.path = path;
        }

        if (left.sum + right.sum + root.val > ans) {
            this.ans = left.sum + right.sum + root.val;
            List<Integer> temp = new ArrayList<>();
            temp.addAll(left.path);
            temp.add(root.val);
            temp.addAll(right.path);
            this.path = temp;
        }
        return new Pair(res, path);
    }

    class Pair {
        int sum;
        List<Integer> path;

        public Pair(int sum, List<Integer> path) {
            this.sum = sum;
            this.path = path;
        }
    }
}