package com.owen.algorithm;

import java.util.*;

/**
 * Created by OKONG on 2020/9/13.
 */
public class ArrayProgramming {
    //[1]两数之和变种(双指针)
    public static List<List<Integer>> twoSum(int[] nums, int start, int target) {
        List<List<Integer>> result = new ArrayList<>();
        int lo = start;
        int hi = nums.length - 1;
        while (lo < hi) {
            int sum = nums[lo] + nums[hi];
            int left = nums[lo];
            int right = nums[hi];
            if (sum == target) {
                List group = new ArrayList<>();
                group.add(left);
                group.add(right);
                result.add(group);
                while (lo < hi && nums[lo] == left) lo++;
                while (lo < hi && nums[hi] == right) hi--;
            } else if (sum > target) {
                while (lo < hi && nums[hi] == right) hi--;
            } else {
                while (lo < hi && nums[lo] == left) lo++;
            }
        }
        return result;
    }

    //[15]三数之和 (双指针)
    public static List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> total = new ArrayList<>();
        for (int i = 0; i < nums.length - 1; ) {
            List<List<Integer>> array = twoSum(nums, i + 1, 0 - nums[i]);
            if (array.size() > 0) {
                for (List<Integer> sum : array) {
                    sum.add(nums[i]);
                }
                total.addAll(array);
            }
            i++;
            while (i < nums.length - 1 && nums[i] == nums[i - 1]) i++;
        }
        return total;
    }

    //[33].搜索旋转排序数组 O(K+ logN)
    public static int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int size = nums.length;
        int k = size - 1;
        int min = nums[k];
        while (k > 0) {
            int pre = nums[k - 1];
            if (min > pre) {
                min = pre;
                k--;
            } else {
                break;
            }
        }

        int start = k, end = k + size - 1;
        while (start <= end) {
            int mid = start + (end - start) / 2;
            if (nums[mid % size] == target) {
                return mid % size;
            } else if (nums[mid % size] > target) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }
        return -1;
    }

    //[33].搜索旋转排序数组 O(logN)
    public static int searchV2(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int size = nums.length;
        int start = 0, end = size - 1;
        while (start <= end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[mid] < nums[end]) {
                //后半部分有序
                if (nums[mid] < target && target <= nums[end]) {
                    start = mid + 1;
                } else {
                    end = mid - 1;
                }
            } else {
                //前半部分有序
                if (nums[start] <= target && target < nums[mid]) {
                    end = mid - 1;
                } else {
                    start = mid + 1;
                }
            }
        }
        return -1;
    }

    //54.螺旋矩阵
    public static List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> result = new ArrayList<>();
        int left = 0, right = matrix[0].length - 1, top = 0, down = matrix.length - 1;
        while (left <= right && top <= down) {
            //从左上到右上
            for (int i = left; i <= right; i++) {
                result.add(matrix[top][i]);
            }
            //从右上到右下
            for (int i = top + 1; i <= down; i++) {
                result.add(matrix[i][right]);
            }
            //从右下到左下，防止被重新遍历
            if (top != down) {
                for (int i = right - 1; i >= left; i--) {
                    result.add(matrix[down][i]);
                }
            }
            //从左下到左上，防止被重新遍历
            if (left != right) {
                for (int i = down - 1; i >= top; i--) {
                    result.add(matrix[i][left]);
                }
            }
            left++;
            right--;
            top++;
            down--;
        }
        return result;
    }

    //[55].跳跃游戏 (贪心)
    public static boolean canJump(int[] nums) {
        //[3,2,1,0,4]
        // 3,3,3,3
        int fast = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            fast = Math.max(fast, i + nums[i]);
            if (fast <= i) return false;
        }
        return true;
    }

    //56.合并区间(贪心)
    public static int[][] merge(int[][] intervals) {
        List<int[]> res = new ArrayList<>();
        int size;
        if (intervals == null || (size = intervals.length) == 0) return res.toArray(new int[0][]);

        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
        //[1,3],[2,6],[4,5],[7,8]

        for (int i = 0; i < size; i++) {
            int start = intervals[i][0];
            int end = intervals[i][1];
            while (i + 1 < size && end >= intervals[i + 1][0]) {
                end = Math.max(end, intervals[i + 1][1]);
                i++;
            }
            res.add(new int[]{start, end});
        }
        return res.toArray(new int[0][]);
    }

    //59.螺旋矩阵II
    public static int[][] generateMatrix(int n) {
        if (n == 0) return null;
        if (n == 1) return new int[][]{{1}};
        int[][] result = new int[n][n];
        int top = 0, down = n - 1, left = 0, right = n - 1, x = 1;
        while (left <= right && top <= down) {
            for (int i = left; i <= right; i++) {
                result[top][i] = x++;
            }
            top++;
            for (int i = top; i <= down; i++) {
                result[i][right] = x++;
            }
            right--;
            if (top != down) {
                for (int i = right; i >= left; i--) {
                    result[down][i] = x++;
                }
                down--;
            }

            if (left != right) {
                for (int i = down; i >= top; i--) {
                    result[i][left] = x++;
                }
                left++;
            }
        }
        return result;
    }

    //68.旋转图像
    public static void rotateMatrix(int[][] matrix) {
        int n = matrix.length;
        if (n == 1) {
            return;
        }

        //规律(i, j) -> (j, n-1-i) -> (n-1-i, n-1-j) ->(n-1-j, i) -> (i, j)
        //(0,0)->(1,1)->(2,2)开始，只需要计算一半
        for (int i = 0; i < n / 2; i++) {
            for (int j = i; j < n - 1 - i; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - j][i];
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j];
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i];
                matrix[j][n - 1 - i] = temp;
            }
        }
    }

    //[73].矩阵置零
    public static void setZeroes(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        //需要刷第一列
        boolean setCol = false;
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0) {
                setCol = true;
            }
            //关键点,映射到第一列上，但不能遍历第一列
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }

        //从[1][1]先覆盖即可
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }

        //第一个曾经是0或者第一行曾经出现过0，覆盖掉第一行
        if (matrix[0][0] == 0) {
            for (int i = 1; i < n; i++) {
                matrix[0][i] = 0;
            }
        }
        //第一列曾经出现过0，覆盖掉第一列
        if (setCol) {
            for (int i = 0; i < m; i++) {
                matrix[i][0] = 0;
            }
        }
    }

    //[74].搜索二维矩阵
    public static boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = m == 0 ? 0 : matrix[0].length;
        int start = 0, end = m * n - 1;
        while (start <= end) {
            int mid = start + (end - start) / 2;
            int i = mid / n, j = mid % n;
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] > target) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }
        return false;
    }

    //[75].颜色分类
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

    //[81].搜索旋转排序数组II
    public static boolean searchII(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        int size = nums.length;
        int start = 0, end = size - 1;
        while (start <= end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] == target) {
                return true;
            }
            if (nums[start] == nums[mid]) {
                start++;
                continue;
            }
            if (nums[start] < nums[mid]) {
                //前半部分有序
                if (nums[start] <= target && target < nums[mid]) {
                    end = mid - 1;
                } else {
                    start = mid + 1;
                }

            } else {
                //后半部分有序
                if (nums[mid] < target && target <= nums[end]) {
                    start = mid + 1;
                } else {
                    end = mid - 1;
                }
            }
        }
        return false;
    }

    //洗牌算法
    void shuffle(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            int rand = new Random().nextInt(nums.length - i + 1) + i;
            int temp = nums[i];
            nums[i] = nums[rand];
            nums[rand] = temp;
        }
    }

    public static void main(String[] args) {
//        [15]三数之和
//        System.out.println(threeSum(new int[]{-1, 1, 2, 11, 0, 1, -2}));
//        System.out.println(threeSum(new int[]{-1, 0, 1, 2, -1, -4}));
//
//        [33].搜索旋转排序数组 O (logN)
//        System.out.println(search(new int[]{4, 5, 6, 7, 0, 1, 2}, 0));
//        System.out.println(search(new int[]{4, 5, 6, 7, 0, 1, 2}, 3));
//        System.out.println(searchV2(new int[]{4, 5, 6, 7, 0, 1, 2}, 0));
//        System.out.println(searchV2(new int[]{4, 5, 6, 7, 0, 1, 2}, 3));
//
//        56.合并区间
//        int[][] fi = new int[][]{{1, 3}, {2, 6}, {4, 5}, {7, 8}};
//        int[][] se = new int[][]{{1, 3}, {2, 6}, {8, 10}, {15, 18}};
//        int[][] one = new int[][]{{1, 3}};
//        merge(se);
//
//        54.螺旋矩阵
//        int[][] matrix = new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
//        int[][] matrix = new int[][]{{1, 2}};
//        System.out.println(spiralOrder(matrix));
//
//        55. 跳跃游戏
//        System.out.println(canJump(new int[]{3, 2, 1, 0, 4}));
//
//        59. 螺旋矩阵II
//        int[][] result = generateMatrix(3);
//
//        68. 旋转图像
//        int[][] one = new int[][]{{1}};
//        int[][] se = new int[][]{{1, 2}, {3, 4}};
//        int[][] th = new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
//        int[][] fo = new int[][]{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
//        int[][] fi = new int[][]{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}};
//        rotateMatrix(fi);
//
//        73. 矩阵置零
//        int[][] a = new int[][]{{1, 1, 1}, {1, 0, 1}, {1, 1, 1}};
//        int[][] b = new int[][]{{0, 1, 2, 0}, {3, 4, 5, 2}, {1, 3, 1, 5}};
//        int[][] c = new int[][]{{1, 0, 1, 1}, {1, 1, 1, 1}, {0, 1, 1, 1}, {1, 1, 0, 1}};
//        int[][] d = new int[][]{{1, 2, 3, 4}, {5, 0, 7, 8}, {0, 10, 11, 12}, {13, 14, 15, 0}};
//        setZeroes(a);
//        setZeroes(b);
//        setZeroes(c);
//        setZeroes(d);
//
//        74. 搜索二维矩阵
//        int[][] matrix = new int[][]{{1, 3, 5, 7}, {10, 11, 16, 20}, {23, 30, 34, 50}};
//        System.out.println(searchMatrix(matrix, 13));
//
//        75. 颜色分类
//        int[] sort = new int[]{};
//        sortColors(sort);
//
//        [81].搜索旋转排序数组II
//        System.out.println(searchII(new int[]{2, 5, 6, 0, 0, 1, 2}, 0));
//        System.out.println(searchII(new int[]{1, 0, 1}, 1));
//        System.out.println(searchII(new int[]{1, 0, 1}, 0));
    }
}
