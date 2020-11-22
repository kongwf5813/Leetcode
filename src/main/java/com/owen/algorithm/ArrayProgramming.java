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

    //[57].插入区间
    public static int[][] insert(int[][] intervals, int[] newInterval) {
        if (newInterval.length == 0) return intervals;
        int left = newInterval[0];
        int right = newInterval[1];
        List<int[]> res = new ArrayList<>();
        for (int i = 0; i < intervals.length; i++) {
            int[] ints = intervals[i];
            if (right < ints[0]) {
                res.add(new int[]{left, right});
                //区间比较小没有重叠，需要更新下区间
                left = ints[0];
                right = ints[1];
            } else if (ints[1] < left) {
                res.add(new int[]{ints[0], ints[1]});
            } else {
                if (ints[0] <= left) {
                    left = ints[0];
                }
                if (right < ints[1]) {
                    right = ints[1];
                }
            }
        }
        //最后的区间需要更新下
        res.add(new int[]{left, right});
        return res.toArray(new int[][]{});
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

    //[179].最大数
    public static String largestNumber(int[] nums) {
        String[] strings = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            strings[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(strings, (a, b) -> (b + a).compareTo(a + b));

        //[0,0]
        if (strings[0].equals("0")) return "0";
        StringBuilder sb = new StringBuilder();
        for (String str : strings) {
            sb.append(str);
        }
        return sb.toString();
    }

    //[209].长度最小的子数组
    public static int minSubArrayLen(int s, int[] nums) {
        int res = Integer.MAX_VALUE;
        int sum = 0;
        for (int p = 0, q = 0; q < nums.length; q++) {
            sum += nums[q];

            //一旦窗口满足，则移动起始节点，直到找到窗口不满足条件为止
            while (sum >= s) {
                res = Math.min(res, q - p + 1);
                sum -= nums[p++];
            }

        }
        return res == Integer.MAX_VALUE ? 0 : res;
    }

    //[215].数组中的第K个最大元素
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


    //[163].多数元素
    public static int majorityElement(int[] nums) {
        int candidate = nums[0];
        int count = 0;
        for (int num : nums) {
            //票数为0，意味着需要替换候选人
            if (0 == count) {
                candidate = num;
            }

            //候选人相同+1
            if (num == candidate) {
                count++;
            } else {
                //候选人不相同，减票数
                count--;
            }
        }
        return candidate;
    }

    //[189].旋转数组
    public static void rotate(int[] nums, int k) {
        int realK = k % nums.length;
        //前n-k个
        swap(nums, 0, nums.length - realK - 1);
        //后k个
        swap(nums, nums.length - realK, nums.length - 1);
        swap(nums, 0, nums.length - 1);
    }

    private static void swap(int[] nums, int x, int y) {
        while (x < y) {
            int temp = nums[y];
            nums[y] = nums[x];
            nums[x] = temp;
            x++;
            y--;
        }
    }

    //[217].存在重复元素
    public static boolean containsDuplicate(int[] nums) {
        Arrays.sort(nums);
        for (int i = 1; i < nums.length; i++) {
            if (nums[i - 1] == nums[i]) return true;
        }
        return false;
    }

    //[219].存在重复元素II
    public static boolean containsNearbyDuplicate(int[] nums, int k) {
        Set<Integer> window = new HashSet<>(k);
        for (int i = 0; i < nums.length; i++) {
            if (window.contains(nums[i])) return true;
            window.add(nums[i]);

            if (window.size() > k) {
                window.remove(nums[i - k]);
            }
        }
        return false;
    }

    //[220].存在重复元素III
    public static boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        TreeSet<Long> set = new TreeSet<>();
        for (int i = 0; i < nums.length; i++) {
            Long ceiling = set.ceiling((long) nums[i]);
            if (ceiling != null && ceiling - (long) nums[i] <= (long) t) return true;

            Long floor = set.floor((long) nums[i]);
            if (floor != null && (long) nums[i] - floor <= (long) t) return true;

            set.add((long) nums[i]);
            if (set.size() > k) {
                set.remove((long) nums[i - k]);
            }
        }
        return false;
    }

    //[228].汇总区间
    public static List<String> summaryRanges(int[] nums) {
        // 输入：nums = [0,2,3,4,6,8,9]
        //输出：["0","2->4","6","8->9"]
        List<String> res = new ArrayList<>();
        if (nums.length == 0) return res;
        if (nums.length == 1) {
            res.add("" + nums[0]);
            return res;
        }

        int left = nums[0], right = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (right + 1 != nums[i]) {
                //添加结果
                if (left == right) res.add("" + left);
                else res.add(left + "->" + right);
                //更新区间
                left = nums[i];
                right = nums[i];
            } else {
                //大1，就更新右区间
                right = nums[i];
            }

            if (i == nums.length - 1) {
                //添加结果
                if (left == right) res.add("" + left);
                else res.add(left + "->" + right);
            }
        }
        return res;
    }

    //[229].求众数II
    public static List<Integer> majorityElement2(int[] nums) {
        List<Integer> res = new ArrayList<>();
        int size = nums.length;
        if (size == 0) return res;

        //超过n/3的票数意味着最多两个众数
        int candidate1 = nums[0];
        int candidate2 = nums[0];
        int count1 = 0, count2 = 0;
        for (int num : nums) {
            //与A相等
            if (candidate1 == num) {
                count1++;
                continue;
            }
            //与B相等
            if (candidate2 == num) {
                count2++;
                continue;
            }

            //如果当前值与AB都不等 且票据为0，则更新候选人
            if (count1 == 0) {
                candidate1 = num;
                count1 = 1;
                continue;
            }
            if (count2 == 0) {
                candidate2 = num;
                count2 = 1;
                continue;
            }

            //当前值与AB都不想等，且票据不为0，不需要更新候选人
            count1--;
            count2--;
        }

        //上一轮遍历找出了两个候选人，但是这两个候选人是否均满足票数大于N/3仍然没法确定，需要重新遍历，确定票数
        count1 = count2 = 0;
        for (int num : nums) {
            if (num == candidate1) {
                count1++;
            } else if (num == candidate2) {
                count2++;
            }
        }

        if (count1 > size / 3) {
            res.add(candidate1);
        }
        if (count2 > size / 3) {
            res.add(candidate2);
        }
        return res;
    }

    //[238].除自身以外数组的乘积
    public static int[] productExceptSelf(int[] nums) {
        //left[i] 定义为0～i-1的前缀乘积
        int[] left = new int[nums.length + 1];
        //right[i]定义为i~n-1的后缀乘积
        int[] right = new int[nums.length + 1];

        left[0] = 1;
        for (int i = 0; i < nums.length; i++) {
            left[i + 1] = left[i] * nums[i];
        }

        right[nums.length] = 1;
        for (int j = nums.length - 1; j >= 0; j--) {
            right[j] = right[j + 1] * nums[j];
        }

        int[] res = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            res[i] = left[i] * right[i + 1];
        }
        return res;
    }

    //[239].滑动窗口最大值
    public static int[] maxSlidingWindow(int[] nums, int k) {
        List<Integer> res = new ArrayList<>();
        LinkedList<Integer> monotonic = new LinkedList<>();
        for (int i = 0; i < nums.length; i++) {
            if (i < k - 1) {
                //先填满k-1个数字
                while (!monotonic.isEmpty() && monotonic.peekLast() < nums[i]) {
                    monotonic.pollLast();
                }
                monotonic.add(nums[i]);
            } else {
                //添加新数字
                while (!monotonic.isEmpty() && monotonic.peekLast() < nums[i]) {
                    monotonic.pollLast();
                }
                monotonic.add(nums[i]);

                //取最大值
                res.add(monotonic.getFirst());

                //移出旧数字
                if (monotonic.getFirst() == nums[i - k + 1]) {
                    monotonic.pollFirst();
                }
            }
        }
        int[] arr = new int[res.size()];
        int i = 0;
        for (Integer num : res) {
            arr[i++] = num;
        }
        return arr;
    }

    //[240].搜索二维矩阵II
    public static boolean searchMatrix2(int[][] matrix, int target) {
        if (matrix == null || matrix.length < 1 || matrix[0].length < 1) return false;
        int row = 0, col = matrix[0].length - 1;
        while (row < matrix.length && col >= 0) {
            if (matrix[row][col] == target) {
                return true;
            } else if (matrix[row][col] < target) {
                row++;
            } else {
                col--;
            }
        }
        return false;
    }

    //[274].H指数
    public static int hIndex(int[] citations) {
        int count = 0;
        Arrays.sort(citations);
        while (count < citations.length && citations[citations.length - 1 - count] > count) {
            count++;
        }
        return count;
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

    //[283].移动零
    public static void moveZeroes(int[] nums) {
        int slow = 0, fast = 0;
        while (fast < nums.length) {
            if (nums[fast] != 0) {
                nums[slow] = nums[fast];
                slow++;
            }
            fast++;
        }
        for (int i = slow; i < nums.length; i++) {
            nums[i] = 0;
        }
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

    //[287].寻找重复数(双指针，判断链表有环)
    public static int findDuplicateV2(int[] nums) {
        int slow = 0, fast = 0;
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        slow = 0;

        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }

    //[324].摆动排序II
    public static void wiggleSort(int[] nums) {
        int size = nums.length;
        //从小到大排列
        Arrays.sort(nums);
        int[] temp = nums.clone();
        for (int i = 0; i < size / 2; i++) {
            //较小区间的最大值
            nums[2 * i] = temp[(size - 1) / 2 - i];
            //较大区间的最大值
            nums[2 * i + 1] = temp[size - 1 - i];
        }
        //奇函数，前面一个没有赋值到temp中
        if (size % 2 == 1) {
            nums[size - 1] = temp[0];
        }
    }

    //[334].递增的三元子序列
    public static boolean increasingTriplet(int[] nums) {
        //最小和次最小
        int min = Integer.MAX_VALUE, secondMin = Integer.MAX_VALUE;
        for (int num : nums) {
            if (num <= min) {
                //更新最小
                min = num;
            } else if (num <= secondMin) {
                //更新次小
                secondMin = num;
            } else {
                //大于最小和次小，则找到了递增子序列
                return true;
            }
        }
        return false;
    }

    //[347].前K个高频元素
    public static int[] topKFrequent(int[] nums, int k) {

        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }

        //用大根堆
        PriorityQueue<Map.Entry<Integer, Integer>> queue = new PriorityQueue<>((a, b) -> b.getValue() - a.getValue());
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            queue.offer(entry);
        }
        int[] res = new int[k];
        for (int i = 0; i < k; i++) {
            res[i] = queue.poll().getKey();
        }
        return res;
    }

    //[383].赎金信
    public static boolean canConstruct(String ransomNote, String magazine) {
        int[] bucket = new int[26];
        for (char ch : magazine.toCharArray()) {
            bucket[ch - 'a']++;
        }

        for (char ch : ransomNote.toCharArray()) {
            if (--bucket[ch - 'a'] < 0) {
                return false;
            }
        }
        return true;
    }

    //[384].打乱数组
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
//        [57].插入区间
//        int[][] res = insert(new int[][]{{1, 2}, {3, 5}, {6, 7}, {8, 10}, {12, 16}}, new int[]{4, 8});
//        int[][] res2 = insert(new int[][]{{1, 3}, {6, 9}}, new int[]{2, 5});
//        int[][] res23 = insert(new int[][]{{1, 3}, {6, 9}}, new int[]{8, 9});
//        int[][] res4 = insert(new int[][]{{6, 7}, {8, 9}}, new int[]{8, 8});
//        int[][] res5 = insert(new int[][]{{1, 1}}, new int[]{});
//        int[][] res6 = insert(new int[][]{{2, 5}, {6, 7}, {8, 9}}, new int[]{0, 1});
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
//        [163].多数元素
//        System.out.println(majorityElement(new int[]{1, 2, 1, 2, 3, 2}));
//
//        [189].旋转数组
//        int[] res = new int[]{1, 2, 3, 4, 5, 6, 7};
//        rotate(res, 3);
//        int[] res = new int[]{-1, -100, 3, 99};
//        rotate(res, 2);
//        System.out.println(Arrays.toString(res));
//
//        [229].求众数II
//        System.out.println(majorityElement2(new int[]{1}));
//        [209].长度最小的子数组
//        System.out.println(minSubArrayLen(7, new int[]{2, 3, 1, 2, 4, 3}));
//        System.out.println(minSubArrayLen(7, new int[]{2, 3, 9, 2, 4, 3}));
//        System.out.println(minSubArrayLen(7, new int[]{}));
//
//        [215].数组中的第K个最大元素
//        System.out.println(findKthLargest(new int[]{3, 2, 1, 5, 6, 4}, 2));
//        System.out.println(findKthLargest(new int[]{3, 2, 3, 1, 2, 4, 5, 5, 6}, 4));
//
//        [217].存在重复元素
//        System.out.println(containsDuplicate(new int[]{1, 2, 3, 1}));
//        System.out.println(containsDuplicate(new int[]{1}));
//        System.out.println(containsDuplicate(new int[]{}));
//
//        [219].存在重复元素II
//        System.out.println(containsNearbyDuplicate(new int[]{1, 2, 3, 1}, 3));
//
//        [220].存在重复元素III
//        System.out.println(containsNearbyAlmostDuplicate(new int[]{1, 2, 3, 1}, 3, 0));
//        System.out.println(containsNearbyAlmostDuplicate(new int[]{1, 5, 9, 1, 5, 9}, 2, 3));
//        System.out.println(containsNearbyAlmostDuplicate(new int[]{-2147483648, 2147483647}, 1, 1));
//
//        [228].汇总区间
//        System.out.println(summaryRanges(new int[]{}));
//        System.out.println(summaryRanges(new int[]{0}));
//        System.out.println(summaryRanges(new int[]{0, 1}));
//        System.out.println(summaryRanges(new int[]{0, 1, 2, 4, 5, 7}));
//        System.out.println(summaryRanges(new int[]{0, 2, 3, 4, 6, 8, 9}));
//
//        [238].除资深以为数组的乘积
//        System.out.println(Arrays.toString(productExceptSelf(new int[]{1, 2, 3, 4})));
//        System.out.println(Arrays.toString(productExceptSelf(new int[]{9, 7})));
//
//        [239].滑动窗口最大值
//        System.out.println(Arrays.toString(maxSlidingWindow(new int[]{1, 3, -1, -3, 5, 3, 6, 7}, 3)));
//        System.out.println(Arrays.toString(maxSlidingWindow(new int[]{1}, 1)));
//
//        [240].搜索二维矩阵IIx
//        int[][] res = new int[][]{{1, 4, 7, 11, 15}, {2, 5, 8, 12, 19}, {3, 6, 9, 16, 22}, {10, 13, 14, 17, 24}, {18, 21, 23, 26, 30}};
//        System.out.println(searchMatrix2(res, 9));
//
//        [274].H指数
//        System.out.println(hIndex(new int[]{3, 0, 6, 1, 5}));
//
//        [275].H指数II
//        System.out.println(hIndex2(new int[]{0, 1, 3, 5, 6}));
//        System.out.println(hIndex2(new int[]{0, 2, 4, 5, 6}));
//
//        [283].移动零
//        moveZeroes(new int[]{0,1,2,3});
//        moveZeroes(new int[]{0,1,0,3,12});
//        moveZeroes(new int[]{1});
//
//        [287].寻找重复数
//        System.out.println(findDuplicate(new int[]{1, 3, 4, 2, 2}));
//        System.out.println(findDuplicateV2(new int[]{1, 3, 4, 2, 2}));
//
//        [324].摆动排序II
//        int[] wiggle1 = new int[]{1,3,2,2,3,1};
//        wiggleSort(wiggle1);
//        System.out.println(Arrays.toString(wiggle1));
//        int[] wiggle2 = new int[]{1, 5, 1, 1, 6, 4};
//        wiggleSort(wiggle2);
//        System.out.println(Arrays.toString(wiggle2));
//        int[] wiggle3 = new int[]{1, 1, 2, 3, 3, 3, 4};
//        wiggleSort(wiggle3);
//        System.out.println(Arrays.toString(wiggle3));
//
//        [347].前K个高频元素
//        System.out.println(Arrays.toString(topKFrequent(new int[] {1,1,1,2,2,3}, 2)));
//        System.out.println(Arrays.toString(topKFrequent(new int[] {1}, 1)));
//
//        [383].赎金信
//        System.out.println(canConstruct("aa", "ab"));
//        System.out.println(canConstruct("aa", "aab"));
//        [384].打乱数组
//        Solution res = new Solution(new int[]{1, 2, 3});
//        System.out.println(Arrays.toString(res.shuffle()));
//        res.reset();
//        System.out.println(Arrays.toString(res.shuffle()));
    }
}
