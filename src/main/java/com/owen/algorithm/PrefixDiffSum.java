package com.owen.algorithm;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class PrefixDiffSum {
    //[523].连续的子数组和
    public static boolean checkSubarraySum(int[] nums, int k) {
        //(preSum[i] - preSum[j]) % k == 0
        //preSum[i] % k == preSum[j] % k
        Map<Integer, Integer> position = new HashMap<>();
        position.put(0, -1);
        int preSum = 0;
        for (int i = 0; i < nums.length; i++) {
            preSum += nums[i];

            int key = k == 0 ? preSum : preSum % k;
            if (position.containsKey(key)) {
                if (i - position.get(key) > 1) {
                    return true;
                }
            } else {
                position.put(key, i);
            }
        }
        return false;
    }

    //[525].连续数组
    public static int findMaxLength(int[] nums) {
        //前缀和，第一次的位置
        Map<Integer, Integer> position = new HashMap<>();
        //用于计算长度的，保证是0，只要出现相等，就能算出最大长度
        position.put(0, -1);

        int res = 0;
        int preSum = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 1) {
                preSum++;
            } else {
                preSum--;
            }

            //再次相等，说明0的个数等于1的个数
            //0000111
            if (position.containsKey(preSum)) {
                res = Math.max(res, i - position.get(preSum));
            } else {
                position.put(preSum, i);
            }
        }
        return res;
    }

    //[532].数组中的k-diff数对
    public static int findPairs(int[] nums, int k) {
        if (k < 0) return 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        int res = 0;
        for (int num : map.keySet()) {
            if (k == 0) {
                if (map.get(num) > 1) {
                    res++;
                }
            } else if (map.containsKey(num + k)) {
                res++;
            }
        }
        return res;
    }


    //[560].和为k的子数组
    public static int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int presum = 0;
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            presum += nums[i];
            if (map.containsKey(presum - k)) {
                count += map.get(presum - k);
            }
            map.put(presum, map.getOrDefault(presum, 0) + 1);
        }
        return count;
    }

    //[1109].航班预定统计
    public static class Solution1109 {
        public int[] diffSum;

        public void createDiffSum(int[] nums) {
            int n = nums.length;
            diffSum = new int[n];
            int temp = 0;
            for (int i = 0; i < n; i++) {
                diffSum[i] = nums[i] - temp;
                temp = nums[i];
            }
        }

        public void insert(int start, int end, int num) {
            diffSum[start] += num;
            if (end + 1 < diffSum.length) {
                diffSum[end + 1] -= num;
            }
        }

        public int[] result() {
            int n = diffSum.length;
            int[] nums = new int[n];
            int temp = 0;
            for (int i = 0; i < diffSum.length; i++) {
                nums[i] = temp + diffSum[i];
                temp = nums[i];
            }
            return nums;
        }

        public int[] corpFlightBookings(int[][] bookings, int n) {
            int[] nums = new int[n];
            Arrays.fill(nums, 0);
            createDiffSum(nums);
            for (int[] book : bookings) {
                insert(book[0] - 1, book[1] - 1, book[2]);
            }
            return result();
        }
    }

    public static void main(String[] args) {
//        [523].连续的子数组和
//        System.out.println(checkSubarraySum(new int[]{23,2,6,4,7}, 6));
//        System.out.println(checkSubarraySum(new int[]{23, 0, 0}, 6));
//
//        [525].连续数组
//        System.out.println(findMaxLength(new int[]{0,0,0,0,1,1,1}));
//
//        [532].数组中的k-diff数对
//        System.out.println(findPairs(new int[]{-1,-2,-3}, 1));
//        System.out.println(findPairs(new int[]{1, 3, 1, 5, 4}, 3));
//        System.out.println(findPairs(new int[]{1,2,4,4,3,3,0,9,2,3}, 3));
//
//        [560].和为k的子数组
//        System.out.println(subarraySum(new int[]{1,1,1}, 2));
//        System.out.println(subarraySum(new int[]{1,2,3}, 3));
//
//        [1109].航班预定统计
//        Solution1109 test = new Solution1109();
//        System.out.println(Arrays.toString(test.corpFlightBookings(new int[][]{{1,2,10},{2,3,20},{2,5,25}}, 5)));

    }

}
