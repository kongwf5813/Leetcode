package com.owen.algorithm;

import java.util.*;

/**
 * Created by OKONG on 2020/9/13.
 */
public class ArrayProgramming {
    //[1].两数之和变种(双指针)
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

    //[15].三数之和 (双指针)
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

    //[54].螺旋矩阵
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

    //[56].合并区间(贪心)
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

    //[59].螺旋矩阵II
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

    //[48].旋转图像
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

    //[80].删除排序数组中的重复项II
    public static int removeDuplicates(int[] nums) {
        int n = nums.length;
        int slow = 0, fast = 1;
        //表示多了几个，== 1，表示有两个重复项
        int count = 0;
        while (fast < n) {
            if (nums[slow] == nums[fast]) {
                count++;
            } else {
                count = 0;
            }

            if (count < 2) {
                //有新的合法值，slow + 1 ，拷贝完，fast + 1
                slow++;
                nums[slow] = nums[fast];
                fast++;
            } else {
                //有重复项，不操作，fast + 1
                fast++;
            }
        }
        //返回的是个数，== index + 1
        return slow + 1;
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

    //[134].加油站
    public static int canCompleteCircuit(int[] gas, int[] cost) {
        int spare = 0;
        int minSpare = Integer.MAX_VALUE;
        int n = gas.length, start = 0;
        for (int i = 0; i < n; i++) {
            //到达i+1总的剩余油量
            spare += gas[i] - cost[i];
            if (spare < minSpare) {
                minSpare = spare;
                start = i;
            }
        }
        return spare >= 0 ? (start + 1) % n : -1;
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

    //[406].根据身高重建队列
    public static int[][] reconstructQueue(int[][] people) {
        //[7,0] [7,1] [6,1] [5,0] [5,2] [4,4]
        //[7,0] [7,1]
        //[7,0] [6,1] [7,1]
        //[5,0] [7,0] [6,1] [7,1]
        //[5,0] [7,0] [5,2] [6,1] [7,1]
        //[5,0] [7,0] [5,2] [6,1] [4,4] [7,1]
        Arrays.sort(people, (p1, p2) -> p1[0] == p2[0] ? p1[1] - p2[1] : p2[0] - p1[0]);
        LinkedList<int[]> res = new LinkedList<>();
        for (int[] each : people) {
            res.add(each[1], each);
        }
        return res.toArray(new int[res.size()][2]);
    }

    //[419].甲板上的战舰
    public static int countBattleships(char[][] board) {
        int m = board.length, n = board[0].length;
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'X' && check(board, i, j)) {
                    res++;
                }
            }
        }
        return res;
    }

    private static boolean check(char[][] board, int i, int j) {
        return !((i >= 1 && board[i - 1][j] == 'X') || (j >= 1 && board[i][j - 1] == 'X'));
    }

    //[435].无重叠区间
    public static int eraseOverlapIntervals(int[][] intervals) {
        if (intervals.length == 0) return 0;

        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));

        int end = intervals[0][1];
        int res = 0;
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] < end) {
                res++;

                end = Math.min(end, intervals[i][1]);
            } else {
                end = intervals[i][1];
            }
        }
        return res;
    }

    //[442].数组中重复的数据
    public static List<Integer> findDuplicates(int[] nums) {
        List<Integer> res = new ArrayList<>();
        int n = nums.length;
        for (int i = n - 1; i >= 0; i--) {
            int index = Math.abs(nums[i]) - 1;
            if (nums[index] < 0) {
                res.add(index + 1);
            }

            nums[index] = -nums[index];

        }
        return res;
    }

    //[447].回旋镖的数量
    public static int numberOfBoomerangs(int[][] points) {
        int res = 0;
        for (int i = 0; i < points.length; i++) {
            Map<Integer, Integer> distanceCount = new HashMap<>();
            for (int j = 0; j < points.length; j++) {
                if (i != j) {
                    int distance = (points[i][0] - points[j][0]) * (points[i][0] - points[j][0]) +
                            (points[i][1] - points[j][1]) * (points[i][1] - points[j][1]);
                    int count = distanceCount.getOrDefault(distance, 0);
                    distanceCount.put(distance, count + 1);
                }
            }
            for (int count : distanceCount.values()) {
                if (count > 1) {
                    res += count * (count - 1);
                }
            }
        }
        return res;
    }

    //[452].用最少数量的箭引爆气球
    public static int findMinArrowShots(int[][] points) {
        Arrays.sort(points, (a, b) -> a[1] > b[1] ? 1 : -1);
        if (points == null || points.length == 0) return 0;
        int res = 1;
        int q = points[0][1];
        for (int i = 1; i < points.length; i++) {
            int start = points[i][0];
            int end = points[i][1];
            if (q < start) {
                q = end;
                //此时更新
                res++;
            }
        }
        return res;
    }

    //[477].汉明距离总和
    public static int totalHammingDistance(int[] nums) {
        //题目是求两两之间的距离
        //转化为计算每一位的汉明距离 == 每一位 1的个数 * 0的个数
        int res = 0;
        for (int i = 0; i < 32; i++) {
            int ones = 0;
            for (int num : nums) {
                if ((num >> i & 1) == 1) {
                    ones++;
                }
            }

            res += ones * (nums.length - ones);
        }
        return res;
    }

    //[495].提莫攻击
    public static int findPoisonedDuration(int[] timeSeries, int duration) {
        int lastEnd = 0, res = 0;
        for (int time : timeSeries) {
            if (time < lastEnd) {
                if (time + duration > lastEnd) {
                    res += time + duration - lastEnd;
                    lastEnd = time + duration;
                }
            } else {
                res += duration;
                lastEnd = time + duration;
            }
        }
        return res;
    }

    //[498].对角线遍历
    public static int[] findDiagonalOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return new int[0];
        }

        int m = matrix.length;
        int n = matrix[0].length;

        int[] res = new int[m * n];
        int direct = 1;
        int row = 0, col = 0, index = 0;

        while (row < m && col < n) {
            res[index++] = matrix[row][col];

            int x = row + (direct == 1 ? -1 : 1);
            int y = col + (direct == 1 ? 1 : -1);

            if (x < 0 || x == m || y < 0 || y == n) {
                if (direct == 1) {
                    //触碰上边界，达到最后一列，往下
                    row += (col == n - 1) ? 1 : 0;
                    //触碰上边界，还没有达到最后一列，往右
                    col += (col < n - 1) ? 1 : 0;
                } else {
                    //触碰左边界，达到最后一行，往右
                    col += (row == m - 1) ? 1 : 0;

                    //触碰左边界，还没有达到最后一行，往下
                    row += (row < m - 1) ? 1 : 0;
                }
                direct = -1 * direct;
            } else {
                row = x;
                col = y;
            }
        }
        return res;
    }

    //[539].最小时间差
    public static int findMinDifference(List<String> timePoints) {
        if (timePoints.size() >= 1440) return 0;
        int[] minutes = new int[timePoints.size()];
        for (int i = 0; i < timePoints.size(); i++) {
            String[] times = timePoints.get(i).split(":");
            int minute = Integer.parseInt(times[0]) * 60 + Integer.parseInt(times[1]);
            minutes[i] = minute;
        }
        Arrays.sort(minutes);
        int min = Integer.MAX_VALUE;
        for (int i = 1; i < minutes.length; i++) {
            min = Math.min(minutes[i] - minutes[i - 1], min);
            if (min == 0) {
                return 0;
            }
        }

        min = Math.min(min, 1440 + minutes[0] - minutes[minutes.length - 1]);
        return min;
    }

    //[540].有序数组中的单一元素
    public static int singleNonDuplicate(int[] nums) {
        int eor = 0;
        for (int n : nums) {
            eor ^= n;
        }
        return eor;
    }

    //[565].数组嵌套
    public static int arrayNesting(int[] nums) {
        //5,4,0,3,1,6,2
        //0-5 5-6 6-2 2-0
        //1-4 4-1
        int res = 0;
        Set<Integer> unique = new HashSet<>();
        for (int i = 0; i < nums.length ; i++) {
            int k = i, count = 0;
            while (!unique.contains(nums[k])) {
                unique.add(nums[k]);
                k = nums[k];
                count++;
            }
            res = Math.max(res, count);
        }
        return res;
    }

    //[575].分糖果
    public static int distributeCandies(int[] candyType) {
        Set<Integer> set = new HashSet<>();
        for (int candy : candyType) {
            set.add(candy);
        }
        return Math.min(set.size(), candyType.length / 2);
    }

    //[598].范围求和II
    public static int maxCount(int m, int n, int[][] ops) {
        for (int[] op: ops) {
            m = Math.min(m, op[0]);
            n = Math.min(n, op[1]);
        }
        return m * n;
    }

    //[611].有效三角形的个数
    public static int triangleNumber(int[] nums) {
        if(nums.length < 3) return 0;

        //2,2,3,4
        Arrays.sort(nums);

        int count = 0;
        for (int k = nums.length -1; k > 1; k --) {
            int i = 0, j = k -1;
            while (i < j) {
                if (nums[i] + nums[j] > nums[k]) {
                    count += j -i;
                    j--;
                } else {
                    i++;
                }
            }
        }
        return count;
    }

    //[621].任务调度器
    public static int leastInterval(char[] tasks, int n) {
        //AAABBBCCCDD  n=2 优先冷却时间的最大值计算 ABCABCABC = 9 再计算实际最大值ABCDABCDABC 11
        int[] cnt = new int[26];
        for (char task : tasks) {
            cnt[task - 'A']++;
        }
        int max = 0;
        for (int count : cnt) {
            max = Math.max(max, count);
        }

        int remain = 0;
        for (int count : cnt) {
            if (count == max) {
                remain++;
            }
        }
        //前半部分为ABCABC值，能配对成功的数量，后半部分为ABC，不需要间隙的数量
        int res = (max -1) * (n+1) + remain;

        return Math.max(res, tasks.length);
    }

    //[628].三个数的最大乘积
    public static int maximumProduct(int[] nums) {
        if (nums.length < 3) return 0;

        Arrays.sort(nums);
        int n = nums.length;
        int res = Math.max(nums[n-1] * nums[n-2] * nums[n-3], nums[n-1] * nums[0] * nums[1]);
        return res;
    }

    //[1288].删除被覆盖区间
    public static int removeCoveredIntervals(int[][] intervals) {
        if (intervals.length == 0) return 0;
        Arrays.sort(intervals, (a, b) ->
                a[0] == b[0] ? b[1] - a[1] : a[0] - b[0]
        );

        int start = intervals[0][0];
        int end = intervals[0][1];
        int covered = 0;
        for (int i = 1; i < intervals.length; i++) {
            int startNew = intervals[i][0];
            int endNew = intervals[i][1];
            //覆盖
            if (start <= startNew && endNew <= end) {
                covered++;
            }
            //相交
            if (end >= startNew && end <= endNew) {
                start = startNew;
                end = endNew;
            }

            //相离
            if (end < startNew) {
                start = startNew;
                end = endNew;
            }
        }
        return intervals.length - covered;
    }

    public static void main(String[] args) {
//        [15].三数之和
//        System.out.println(threeSum(new int[]{-1, 1, 2, 11, 0, 1, -2}));
//        System.out.println(threeSum(new int[]{-1, 0, 1, 2, -1, -4}));
//
//        [33].搜索旋转排序数组 O (logN)
//        System.out.println(search(new int[]{4, 5, 6, 7, 0, 1, 2}, 0));
//        System.out.println(search(new int[]{4, 5, 6, 7, 0, 1, 2}, 3));
//        System.out.println(searchV2(new int[]{4, 5, 6, 7, 0, 1, 2}, 0));
//        System.out.println(searchV2(new int[]{4, 5, 6, 7, 0, 1, 2}, 3));
//
//        [56].合并区间
//        int[][] fi = new int[][]{{1, 3}, {2, 6}, {4, 5}, {7, 8}};
//        int[][] se = new int[][]{{1, 3}, {2, 6}, {8, 10}, {15, 18}};
//        int[][] one = new int[][]{{1, 3}};
//        merge(se);
//
//        [54].螺旋矩阵
//        int[][] matrix = new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
//        int[][] matrix = new int[][]{{1, 2}};
//        System.out.println(spiralOrder(matrix));
//
//        [55].跳跃游戏
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
//        [59].螺旋矩阵II
//        int[][] result = generateMatrix(3);
//
//        [68].旋转图像
//        int[][] one = new int[][]{{1}};
//        int[][] se = new int[][]{{1, 2}, {3, 4}};
//        int[][] th = new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
//        int[][] fo = new int[][]{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
//        int[][] fi = new int[][]{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}};
//        rotateMatrix(fi);
//
//        [73].矩阵置零
//        int[][] a = new int[][]{{1, 1, 1}, {1, 0, 1}, {1, 1, 1}};
//        int[][] b = new int[][]{{0, 1, 2, 0}, {3, 4, 5, 2}, {1, 3, 1, 5}};
//        int[][] c = new int[][]{{1, 0, 1, 1}, {1, 1, 1, 1}, {0, 1, 1, 1}, {1, 1, 0, 1}};
//        int[][] d = new int[][]{{1, 2, 3, 4}, {5, 0, 7, 8}, {0, 10, 11, 12}, {13, 14, 15, 0}};
//        setZeroes(a);
//        setZeroes(b);
//        setZeroes(c);
//        setZeroes(d);
//
//        [74].搜索二维矩阵
//        int[][] matrix = new int[][]{{1, 3, 5, 7}, {10, 11, 16, 20}, {23, 30, 34, 50}};
//        System.out.println(searchMatrix(matrix, 13));
//
//        [75].颜色分类
//        int[] sort = new int[]{};
//        sortColors(sort);
//
//        [80].删除排序数组中的重复项II
//        System.out.println(removeDuplicates(new int[]{0,0,1,1,1,1,2,3,3}));
//        System.out.println(removeDuplicates(new int[]{1,1,1,2,2,3}));
//
//        [81].搜索旋转排序数组II
//        System.out.println(searchII(new int[]{2, 5, 6, 0, 0, 1, 2}, 0));
//        System.out.println(searchII(new int[]{1, 0, 1}, 1));
//        System.out.println(searchII(new int[]{1, 0, 1}, 0));
//
//        [134].加油站
//        System.out.println(canCompleteCircuit(new int[]{1,2,3,4,5}, new int[]{3,4,5,1,2}));
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
//        [240].搜索二维矩阵IIx
//        int[][] res = new int[][]{{1, 4, 7, 11, 15}, {2, 5, 8, 12, 19}, {3, 6, 9, 16, 22}, {10, 13, 14, 17, 24}, {18, 21, 23, 26, 30}};
//        System.out.println(searchMatrix2(res, 9));
//
//        [274].H指数
//        System.out.println(hIndex(new int[]{3, 0, 6, 1, 5}));
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
//
//        [384].打乱数组
//        Solution res = new Solution(new int[]{1, 2, 3});
//        System.out.println(Arrays.toString(res.shuffle()));
//        res.reset();
//        System.out.println(Arrays.toString(res.shuffle()));
//
//        [406].根据身高重建队列
//        int[][] res = reconstructQueue(new int[][]{{7,0},{4,4},{7,1},{5,0},{6,1},{5,2}});
//
//        [419].甲板上的战舰
//        System.out.println(countBattleships(new char[][]{{'X', '.', '.', 'X'}, {'.', '.', '.', 'X'}, {'.', '.', '.', 'X'}}));
//
//        [435].无重叠区间
//        System.out.println(eraseOverlapIntervals(new int[][]{{1, 2}, {2, 3}, {3, 4}, {1, 3}}));
//        System.out.println(eraseOverlapIntervals(new int[][]{{1, 2}, {1, 2}, {1, 2}}));
//        System.out.println(eraseOverlapIntervals(new int[][]{{1, 2}, {2, 3}}));
//        System.out.println(eraseOverlapIntervals(new int[][]{{1, 2}, {2, 3},{3,4},{-100,-2},{5,7}}));
//
//        [442].数组中重复的数据
//        System.out.println(findDuplicates(new int[]{2, 3, 4, 3}));
//
//        [447].回旋镖的数量
//        System.out.println(numberOfBoomerangs(new int[][]{{0,0},{1,1},{2,2}}));
//        System.out.println(numberOfBoomerangs(new int[][]{{0,0}}));
//        System.out.println(numberOfBoomerangs(new int[][]{{0,0}, {1,1}}));
//
//        [452].用最少数量的箭引爆气球
//        System.out.println(findMinArrowShots(new int[][]{{1, 2}, {2, 3}, {3, 4}, {4, 5}}));
//        System.out.println(findMinArrowShots(new int[][]{{1,2},{3,4},{5,6},{7,8}}));
//        System.out.println(findMinArrowShots(new int[][]{{-2147483646,-2147483645},{2147483646,2147483647}}));
//
//        [477].汉明距离总和
//        System.out.println(totalHammingDistance(new int[]{4, 14, 2}));
//
//        [495].提莫攻击
//        System.out.println(findPoisonedDuration(new int[]{1,2,3,4,5}, 5));
//        System.out.println(findPoisonedDuration(new int[]{1,2,3,4}, 0));
//
//        [498].对角线遍历
//        System.out.println(Arrays.toString(findDiagonalOrder(new int[][]{{1, 2, 3}, { 4, 5, 6}, {7, 8, 9}})));
//
//        [539].最小时间差
//        System.out.println(findMinDifference(Arrays.asList("23:59","00:00")));
//
//        [540]. 有序数组中的单一元素
//        System.out.println(singleNonDuplicate(new int[]{3,3,7,7,10,11,11}));
//
//        [565].数组嵌套
//        System.out.println(arrayNesting(new int[]{5,4,0,3,1,6,2}));
//        System.out.println(arrayNesting(new int[]{0}));
//
//        [575].分糖果
//        System.out.println(distributeCandies(new int[]{1,1,2,3}));
//
//        [611].有效三角形的个数
//        System.out.println(triangleNumber(new int[]{2,2,3,4}));
//
//        [621].任务调度器
//        System.out.println(leastInterval(new char[]{'A', 'A', 'A', 'B', 'B', 'B'}, 2));
//        System.out.println(leastInterval(new char[]{'A','A','A','A','A','A','B','C','D','E','F','G'}, 2));
//
//        [628].三个数的最大乘积
//        System.out.println(maximumProduct(new int[] {-3, -2, 1,3}));
//
//        [1288].删除被覆盖区间
//        System.out.println(removeCoveredIntervals(new int[][] {{1,4},{3,6},{2,8}}));
    }
}
