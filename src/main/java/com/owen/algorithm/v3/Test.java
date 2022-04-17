package com.owen.algorithm.v3;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.owen.algorithm.v3.AllOfThem.*;

import org.apache.commons.lang3.builder.Diff;


public class Test {


    public static void main(String[] args) {
//        System.out.println(numSquares(7));

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
            if(j + 1 < diff.length) {
                diff[j+1] -= value;
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
}