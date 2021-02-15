package com.owen.algorithm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import chemaxon.jep.function.In;

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

    //[153].寻找旋转排序数组中的最小值
    public static int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                //最小值，可能mid就是最小的，所以不能-1。
                //例如3, 1, 2
                right = mid;
            }
        }
        return nums[left];
    }

    //[154].寻找旋转排序数组中的最小值II
    public static int findMin2(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else if (nums[mid] < nums[right]) {
                right = mid;
            } else {
                //砍掉右侧边界
                //101111, 1110111
                right--;
            }
        }
        return nums[left];
    }

    //[162].寻找峰值
    public static int findPeakElement(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[mid + 1]) {
                right = mid;
            } else {
                //不存在nums[mid] = nums[mid + 1]，肯定是小于
                left = mid + 1;
            }
        }
        return left;
    }

    //[275].H指数II
    public static int hIndex2(int[] citations) {
        int n = citations.length;
        int left = 0, right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (n - mid == citations[mid]) {
                return n - mid;
            } else if (n - mid < citations[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return n - left;
    }

    //[287].寻找重复数(二分搜索)
    public static int findDuplicate(int[] nums) {
        //题目是从1到n的数字,一共n+1个数
        int n = nums.length - 1;
        int left = 1, right = n;
        while (left < right) {
            int mid = left + (right - left) / 2;
            int count = 0;
            for (int i = 0; i <= n; i++) {
                if (nums[i] <= mid) {
                    count++;
                }
            }

            //1,3,4,2,2, mid = 2， count = 3， 说明重复数在区间[1,2]
            if (count > mid) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
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

    public static int kthSmallest(int[][] matrix, int k) {
        int n = matrix.length;
        int left = matrix[0][0], right = matrix[n - 1][n - 1];
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (check(matrix, k, mid, n)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    private static boolean check(int[][] matrix, int k, int mid, int n) {
        int x = n - 1;
        int y = 0;
        int num = 0;
        while (x >= 0 && y < n) {
            //从左下角开始统计
            if (matrix[x][y] <= mid) {
                //统计比mid小的总数
                num += x + 1;
                y++;
            } else {
                x--;
            }
        }
        //如果比mid小的总数大于等于k,意味着需要右边界需要缩小范围
        return num >= k;
    }

    //[436].寻找右区间
    public static int[] findRightInterval(int[][] intervals) {
        int n = intervals.length;
        int[][] sorted = new int[n][2];
        for (int i = 0; i < n; i++) {
            sorted[i] = new int[]{intervals[i][0], i};
        }
        Arrays.sort(sorted, (a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);

        int[] res = new int[n];
        for (int i = 0; i < n; i++) {
            int l = 0, r = n - 1;

            while (l < r) {
                int mid = l + (r - l) / 2;
                if (sorted[mid][0] < intervals[i][1]) {
                    l = mid + 1;
                } else if (sorted[mid][0] > intervals[i][1]) {
                    r = mid;
                } else {
                    r = mid;
                }
            }
            //确实大于，返回坐标，否则返回-1
            res[i] = sorted[l][0] >= intervals[i][1] ? sorted[l][1] : -1;
        }
        return res;
    }

    //[475].供暖器
    public static int findRadius(int[] houses, int[] heaters) {
        Arrays.sort(heaters);
        int maxRadius = Integer.MIN_VALUE;
        for (int house : houses) {
            maxRadius = Math.max(searchHeater(heaters, house), maxRadius);
        }
        return maxRadius;
    }

    private static int searchHeater(int[] heaters, int house) {
        int left = 0, right = heaters.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (heaters[mid] == house) {
                return 0;
            } else if (house < heaters[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        if (left > heaters.length - 1) {
            return house - heaters[heaters.length - 1];
        } else if (right < 0) {
            return heaters[0] - house;
        } else {
            //后面left比right大
            return Math.min(heaters[left] - house, house - heaters[right]);
        }
    }

    //[875].爱吃香蕉的珂珂
    public static int minEatingSpeed(int[] piles, int H) {
        int left = 1, right = Arrays.stream(piles).max().getAsInt();
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (eatingTime(piles, mid) <= H) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    //给定一次吃香蕉的个数，求得的时间
    private static int eatingTime(int[] piles, int k) {
        int res = 0;
        for (int pile : piles) {
            res += pile / k + (pile % k > 0 ? 1 : 0);
        }
        return res;
    }

    //[1011].在D天内送达包裹的能力
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

    public static void main(String[] args) {
//        [34].在排序数组中查找元素的第一个和最后一个的位置
//        int[] result2 = searchRange(new int[]{5, 7, 7, 8, 8, 10}, 6);
//        int[] result3 = searchRange(new int[]{5, 7, 7, 8, 8, 10}, 7);
//        int[] result4 = searchRange(new int[]{5, 7, 7, 8, 8, 10}, 5);
//        int[] result8 = searchRange(new int[]{}, 10);
//
//        [50].Pow(x, n)
//        System.out.println(myPow(2.00000d, 10));
//        System.out.println(myPow(2.10000d, 3));
//        System.out.println(myPow(2.00000d, -2));
//
//        [153].寻找旋转排序数组中的最小值
//        System.out.println(findMin(new int[]{4, 5, 6, 7, 0, 1, 2}));
//        System.out.println(findMin(new int[]{3, 4, 5, 1, 2}));
//        System.out.println(findMin(new int[]{0, 1, 2, 3, 4}));
//        System.out.println(findMin(new int[]{3, 1, 2}));
//        System.out.println(findMin(new int[]{4, 1, 2, 3}));
//
//        [154].寻找旋转排序数组中的最小值II
//        System.out.println(findMin2(new int[]{2, 2, 0, 1, 2}));
//        System.out.println(findMin2(new int[]{0, 1, 0}));
//        System.out.println(findMin2(new int[]{1, 0, 1, 1, 1, 1}));
//
//        [162].寻找峰值
//        System.out.println(findPeakElement(new int[]{1, 2, 3}));
//        System.out.println(findPeakElement(new int[]{1,2,3,1}));
//        System.out.println(findPeakElement(new int[]{1,2,1,3,5,6,4}));
//
//        [275].H指数II
//        System.out.println(hIndex2(new int[]{0, 1, 3, 5, 6}));
//        System.out.println(hIndex2(new int[]{0, 2, 4, 5, 6}));
//
//        [287].寻找重复数
//        System.out.println(findDuplicate(new int[]{1, 3, 4, 2, 2}));
//
//        [436].寻找右区间
//        System.out.println(Arrays.toString(findRightInterval(new int[][]{{3,4},{2,3},{1,2}})));
//
//        [475].供暖器
//        System.out.println(findRadius(new int[]{1,5}, new int[]{2}));
//        System.out.println(findRadius(new int[]{1,2,3,4}, new int[]{1,4}));
//
//        [875].爱吃香蕉的珂珂
//        System.out.println(minEatingSpeed(new int[]{3,6,7,11}, 8));
//
//        [1011].在D天内送达包裹的能力
//        System.out.println(shipWithinDays(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 5));
//        System.out.println(shipWithinDays(new int[]{3, 2, 2, 4, 1, 4}, 3));
//        System.out.println(shipWithinDays(new int[]{337, 399, 204, 451, 273, 471, 37, 211, 67, 224, 126, 123, 294, 295, 498, 69, 264, 307, 419, 232, 361, 301, 116, 216, 227, 203, 456, 195, 444, 302, 58, 496, 84, 280, 58, 107, 300, 334, 418, 241}, 20));
    }

}
