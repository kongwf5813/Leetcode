package com.owen.algorithm;

import java.util.Arrays;

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

    //[875].爱吃香蕉的珂珂
    public static int minEatingSpeed(int[] piles, int H) {
        int left = 1, right = Arrays.stream(piles).max().getAsInt();
        while (left < right) {
            int mid = left + (right - left) /2;
            if (eatingTime(piles, mid) <= H) {
                right = mid;
            } else {
                left = mid +1;
            }
        }
        return left;
    }

    //给定一次吃香蕉的个数，求得的时间
    private static int eatingTime(int[] piles, int k) {
        int res = 0;
        for (int pile :piles) {
            res += pile / k + (pile %k > 0 ? 1: 0);
        }
        return res;
    }

    //[1011]在D天内送达包裹的能力
    public static int shipWithinDays(int[] weights, int D) {
        int left = Arrays.stream(weights).max().getAsInt(), right = Arrays.stream(weights).sum();
        while (left < right) {
            int mid = left + (right-left)/2;
            if (needDays(weights, mid) <= D) {
                right = mid;
            } else {
                left = mid +1;
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

//        System.out.println(minEatingSpeed(new int[]{3,6,7,11}, 8));

        System.out.println(shipWithinDays(new int[] {1,2,3,4,5,6,7,8,9,10}, 5));
        System.out.println(shipWithinDays(new int[] {3,2,2,4,1,4}, 3));
        System.out.println(shipWithinDays(new int[] {337,399,204,451,273,471,37,211,67,224,126,123,294,295,498,69,264,307,419,232,361,301,116,216,227,203,456,195,444,302,58,496,84,280,58,107,300,334,418,241}, 20));
    }

}
