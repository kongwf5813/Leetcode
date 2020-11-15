package com.owen.algorithm;

/**
 * Created by OKONG on 2020/9/13.
 */
public class BinarySearch {

    //二分搜索
    int binary_search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            }
        }
        return -1;
    }

    //寻找左侧边界的二分查找
    int leftbound_search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                //右侧边界往左边缩
                right = mid - 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            }
        }
        if (left >= nums.length || nums[left] != target) {
            return -1;
        }
        return left;
    }

    //寻找右侧边界的二分查找
    int rightbound_search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                //左侧边界往右边扩
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            }
        }
        if (right < 0 || nums[right] != target) {
            return -1;
        }
        return right;
    }

    //[34].在排序数组中查找元素的第一个和最后一个位置
    public static int[] searchRange(int[] nums, int target) {
        int leftIndex = findIndex(nums, target, true);
        int rightIndex = findIndex(nums, target, false);
        return new int[]{leftIndex, rightIndex};
    }

    private static int findIndex(int[] nums, int target, boolean left) {
        if (nums.length == 0) return -1;
        int start = 0, end = nums.length - 1;
        while (start <= end) {
            int mid = start + (end - start) / 2;
            if (target == nums[mid]) {
                //寻找左边界
                if (left) {
                    end = mid - 1;
                } else {
                    //寻找右边界
                    start = mid + 1;
                }
            } else if (target > nums[mid]) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }
        //一直搜大的没搜到，一直搜小没搜到，start就是
        if (left && (start >= nums.length || nums[start] != target)) return -1;
        if (!left && (end < 0 || nums[end] != target)) return -1;
        return left ? start : end;
    }

    //50 Pow(x,n)
    public static double myPow(double x, int n) {
        if (n == 0) return 1;
        if (n == 1) return x;
        if (n == -1) return 1 / x;

        double half = myPow(x, n / 2);
        double rest = myPow(x, n % 2);
        return half * half * rest;
    }


    //[374].猜数字大小
    public class Solution {
        public int guessNumber(int n) {
            int left = 1, right = n;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                int res = guess(mid);
                if (res == 0) {
                    return mid;
                } else if (res < 0) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            return -1;
        }

        private int guess(int mid) {
            return 0;
        }
    }

    public static void main(String[] args) {
//        34. 在排序数组中查找元素的第一个和最后一个的位置
//        int[] result2 = searchRange(new int[]{5, 7, 7, 8, 8, 10}, 6);
//        int[] result3 = searchRange(new int[]{5, 7, 7, 8, 8, 10}, 7);
//        int[] result4 = searchRange(new int[]{5, 7, 7, 8, 8, 10}, 5);
//        int[] result8 = searchRange(new int[]{}, 10);
//
//        50. Pow(x, n)
//        System.out.println(myPow(2.00000d, 10));
//        System.out.println(myPow(2.10000d, 3));
//        System.out.println(myPow(2.00000d, -2));

    }

}
