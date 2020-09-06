package com.owen;

import java.util.*;

public class AlgorithmMananger {

    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    //[2]两数相加
    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode output = new ListNode(0);
        ListNode prev = output;
        //进位标志
        int carry = 0;
        //只要下一位有数据就前进
        while (l1 != null || l2 != null) {
            int value = 0;
            if (l1 != null) {
                value += l1.val;
                l1 = l1.next;
            }

            if (l2 != null) {
                value += l2.val;
                l2 = l2.next;
            }
            value += carry;
            //prev.next才是当前节点
            prev.next = new ListNode(value % 10);
            prev = prev.next;

            //下一位的进位数据
            carry = value / 10;
        }
        if (carry != 0) {
            prev.next = new ListNode(carry);
        }
        //第一个节点直接被忽略掉
        return output.next;
    }

    //[3]无重复子串的最长子串
    public static int lengthOfLongestSubstring(String s) {
        //滑动窗口
        int[] position = new int[128];
        int start = 0, end = 0;
        int result = 0;
        char[] array = s.toCharArray();
        while (end < array.length) {
            char character = array[end];
            int lastMaxIndex = position[character];
            //滑动窗口缩小
            start = Math.max(start, lastMaxIndex);
            //更新最长子串的长度
            result = Math.max(result, end - start + 1);
            //更新字符的最大位置
            position[character] = end + 1;
            //滑动窗口扩大
            end++;
        }
        return result;
    }

    //[5]最长回文子串
    public static String longestPalindrome(String s) {
        int size = s.length();
        //如果求子串，那么boolean dp[i][j]的定义: i到j是否是回文子串
        //如果求最大长度，那么int dp[i][j]定义: i到j范围内最大长度
        boolean[][] dp = new boolean[size][size];
        for (int i = 0; i < size; i++) {
            dp[i][i] = true;
        }
        int maxLen = size == 0 ? 0 : 1;
        int start = 0;
        char[] chars = s.toCharArray();
        for (int i = size - 1; i >= 0; i--) {
            for (int j = i + 1; j < size; j++) {
                dp[i][j] = (chars[i] == chars[j]) && (j - i < 3 || dp[i + 1][j - 1]);
                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    start = i;
                }
            }
        }
        return s.substring(start, start + maxLen);
    }

    public static String longestPalindromeV2(String s) {
        String res = "";
        for (int i = 0; i < s.length(); i++) {
            String s1 = palindrome(s, i, i);
            String s2 = palindrome(s, i, i + 1);
            res = s1.length() > res.length() ? s1 : res;
            res = s2.length() > res.length() ? s2 : res;

        }
        return res;
    }

    private static String palindrome(String s, int l, int r) {
        while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
            l--;
            r++;
        }
        return s.substring(l + 1, r);
    }

    //[15]三数之和
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

    //[1]两数之和变种
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

    //[53]最大子序列和
    public static int maxSubArray(int[] nums) {
        int size = nums.length;
        if (size == 0) return 0;
        int dp0 = nums[0];
        int dp1;
        int res = dp0;
        for (int i = 1; i < size; i++) {
            //-3 4 1 i=1的时候代表只有4作为子数组
            //如果前面的加上本身比本身大，那么考虑加入子数组
            dp1 = Math.max(nums[i], nums[i] + dp0);

            //最大值
            res = Math.max(res, dp1);
            dp0 = dp1;
        }
        return res;
    }

    //[91]解码方法
    public static int numDecodings(String s) {
        if (s.length() == 0 || s.charAt(0) == '0') return 0;
        int dp[] = new int[s.length()];
        dp[0] = 1;
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) == '0') {
                if (s.charAt(i - 1) == '0' || s.charAt(i - 1) > '2') {
                    return 0;
                } else if (i > 1) {
                    dp[i] = dp[i - 2];
                } else {
                    dp[i] = 1;
                }
            } else if (s.charAt(i - 1) == '1' || (s.charAt(i - 1) == '2' && '1' <= s.charAt(i) && s.charAt(i) <= '6')) {
                if (i > 1) {
                    dp[i] = dp[i - 1] + dp[i - 2];
                } else {
                    dp[i] = dp[i - 1] + 1;
                }
            } else {
                dp[i] = dp[i - 1];
            }
        }
        return dp[s.length() - 1];
    }

    //[46]全排列
    public static List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new LinkedList<>();
        LinkedList<Integer> tack = new LinkedList<>();
        backForPermute(nums, tack, res);
        return res;
    }

    private static void backForPermute(int[] nums, LinkedList<Integer> select, List<List<Integer>> res) {
        //结束条件
        if (nums.length == select.size()) {
            res.add(new LinkedList<>(select));
            return;
        }

        //决策树遍历
        for (int i = 0; i < nums.length; i++) {
            //排除不合法的选择
            if (select.contains(nums[i])) {
                continue;
            }
            //做选择
            select.add(nums[i]);
            //进入下一层决策
            backForPermute(nums, select, res);
            //撤销选择
            select.removeLast();
        }
    }

    //51. N皇后
    public static List<List<String>> solveNQueens(int n) {
        List<String> select = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            sb.append('.');
        }
        for (int i = 0; i < n; i++) {
            select.add(sb.toString());
        }
        List<List<String>> solutions = new ArrayList<>();
        trackForNQueens(0, select, solutions);
        return solutions;
    }

    private static void trackForNQueens(int row, List<String> select, List<List<String>> solutions) {
        if (row == select.size()) {
            solutions.add(new ArrayList<>(select));
            return;
        }
        char[] value = select.get(row).toCharArray();
        for (int col = 0; col < value.length; col++) {
            if (!isValid(select, row, col)) {
                continue;
            }

            value[col] = 'Q';
            select.set(row, new String(value));
            trackForNQueens(row + 1, select, solutions);
            value[col] = '.';
            select.set(row, new String(value));
        }
    }

    private static boolean isValid(List<String> select, int row, int col) {
        int n = select.get(row).length();
        //因为是从第0行的每列开始的，所以不需要判断左下，右下的情况
        //判断列不合法
        for (int i = 0; i < n; i++) {
            if ('Q' == select.get(i).charAt(col)) {
                return false;
            }
        }
        //判断左上
        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
            if ('Q' == select.get(i).charAt(j)) {
                return false;
            }
        }

        //判断右上
        for (int i = row, j = col; i >= 0 && j < n; i--, j++) {
            if ('Q' == select.get(i).charAt(j)) {
                return false;
            }
        }
        return true;
    }

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

    //洗牌算法
    void shuffle(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            int rand = new Random().nextInt(nums.length - i + 1) + i;
            int temp = nums[i];
            nums[i] = nums[rand];
            nums[rand] = temp;
        }
    }

    //37.解数独
    public void solveSudoku(char[][] board) {
        if (sudokuBackTrace(board, 0, 0)) {
            System.out.println(board);
        } else {
            System.out.println("无解");
        }
    }

    private static boolean sudokuBackTrace(char[][] board, int i, int j) {
        if (j == 9) {
            return sudokuBackTrace(board, i + 1, 0);
        }

        if (i == 9) {
            return true;
        }

        //如果是数字已经填充了，继续跳过
        if (board[i][j] != '.') {
            return sudokuBackTrace(board, i, j + 1);
        }

        for (char ch = '1'; ch <= '9'; ch++) {
            if (!isValid(board, i, j, ch)) {
                continue;
            }

            board[i][j] = ch;
            if (sudokuBackTrace(board, i, j + 1)) {
                return true;
            }
            board[i][j] = '.';

        }
        return false;
    }

    private static boolean isValid(char[][] board, int r, int c, char ch) {
        for (int k = 0; k < 9; k++) {
            if (board[r][k] == ch) return false;
            if (board[k][c] == ch) return false;
            if (board[(r / 3) * 3 + k / 3][(c / 3) * 3 + k % 3] == ch) return false;
        }
        return true;
    }

    //22. 括号生成
    public static List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        if (n <= 0) return result;
        backtraceParenthesis(new StringBuilder(), n, n, result);
        return result;
    }

    private static void backtraceParenthesis(StringBuilder sb, int left, int right, List<String> whole) {
        //结束条件
        if (right < left || left < 0 || right < 0) {
            return;
        }

        //都选完了，说明产生了一个合法解
        if (left == 0 && right == 0) {
            whole.add(sb.toString());
            return;
        }

        sb.append('(');
        backtraceParenthesis(sb, left - 1, right, whole);
        sb.deleteCharAt(sb.length() - 1);

        sb.append(')');
        backtraceParenthesis(sb, left, right - 1, whole);
        sb.deleteCharAt(sb.length() - 1);
    }

    //19.删除链表的倒数第N个节点
    public static ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode fast = head, slow = head, pre = null;
        int i = 0;
        while (i++ < n) {
            fast = fast.next;
        }
        while (fast != null) {
            pre = slow;
            fast = fast.next;
            slow = slow.next;
        }

        if (pre == null) {
            pre = slow.next;
            return pre;
        } else {
            pre.next = slow.next;
            return head;
        }
    }

    //TODO 黄金矿工
    public static int getMaximumGold(int[][] grid) {
        return 0;
    }

    //25.K个一组翻转链表
    public static ListNode reverseKGroup(ListNode head, int k) {
        if (head == null) return null;
        ListNode start = head, end = head;
        for (int i = 0; i < k; i++) {
            //k不足返回的是头
            if (end == null) return head;
            end = end.next;
        }

        ListNode newHead = reverse(start, end);
        start.next = reverseKGroup(end, k);
        return newHead;
    }

    //206. 反转链表迭代
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

    //206.反转链表递归
    public static ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode last = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return last;
    }

    //61.旋转链表
    public static ListNode rotateRight(ListNode head, int k) {
        ListNode cur = head, last = null;
        int n = 0;
        while (cur != null) {
            last = cur;
            cur = cur.next;
            n++;
        }
        int realK;
        if (n == 1 || n == 0 || (realK = k % n) == 0) {
            return head;
        }

        //双指针
        ListNode fast = head, slow = head, start = null;
        int i = 0;
        while (i++ < realK) {
            fast = fast.next;
        }

        while (fast != null) {
            start = slow;
            fast = fast.next;
            slow = slow.next;
        }
        start.next = null;
        last.next = head;
        return slow;
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

    //56.合并区间
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

    //62.不同路径 动态规划
    public static int uniquePaths(int m, int n) {
        if (m == 0 || n == 0) return 0;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 0; i < n; i++) {
            dp[0][i] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i][j - 1] + dp[i - 1][j];
            }
        }
        return dp[m - 1][n - 1];
    }

    //63.不同路径II
    public static int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m, n;
        if ((m = obstacleGrid.length) == 0 || (n = obstacleGrid[0].length) == 0) return 0;
        int[][] dp = new int[m][n];

        for (int i = 0; i < m; i++) {
            if (obstacleGrid[i][0] == 0) {
                dp[i][0] = 1;
            } else {
                break;
            }
        }
        for (int i = 0; i < n; i++) {
            if (obstacleGrid[0][i] == 0) {
                dp[0][i] = 1;
            } else {
                break;
            }
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = dp[i][j - 1] + dp[i - 1][j];
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    //21.合并两个有序列表
    public static ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode prehead = new ListNode(-1);
        ListNode pre = prehead;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                pre.next = l1;
                l1 = l1.next;
            } else {
                pre.next = l2;
                l2 = l2.next;
            }
            pre = pre.next;
        }
        pre.next = l1 == null ? l2 : l1;
        return prehead.next;
    }

    //[60].第k个排列
    public static String getPermutation(int n, int k) {
        int[] facto = new int[n + 1];
        facto[0] = 1;
        List<Integer> select = new ArrayList<>(k);
        for (int i = 1; i <= n; i++) {
            facto[i] = facto[i - 1] * i;
            select.add(i);
        }

        --k;//选择的范围是从0开始，所以需要减1
        StringBuilder sb = new StringBuilder();
        for (int j = n - 1; j >= 0; j--) {
            //k/(n-1)!, 算出第几个数
            int index = k / facto[j];
            sb.append(select.remove(index));

            k -= index * facto[j];
        }
        return sb.toString();
    }

    //[49] 字母异位分组
    public static List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> groupMap = new HashMap<>();
        for (String str : strs) {
            char[] sorted = str.toCharArray();
            Arrays.sort(sorted);
            String key = String.valueOf(sorted);
            List<String> value = groupMap.get(key);
            if (value == null) {
                value = new ArrayList<>();
                groupMap.put(key, value);
            }
            value.add(str);
        }
        List<List<String>> result = new ArrayList<>();
        for (List<String> val : groupMap.values()) {
            result.add(val);
        }
        return result;
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

    //28. 实现strStr()
    public static int strStr(String haystack, String needle) {
        if (needle == null || needle.length() == 0) {
            return 0;
        }
        if (haystack == null || haystack.length() == 0 || needle.length() > haystack.length()) {
            return -1;
        }
        int l = 0;
        int r = 0;
        while (l < haystack.length() && r < needle.length()) {
            if (needle.charAt(r) == haystack.charAt(l)) {
                r++;
                l++;
            } else {
                l = l - r + 1;
                r = 0;
            }
        }
        if (r == needle.length()) {
            return l - needle.length();
        } else {
            return -1;
        }
    }

    //64. 最小路径和
    public static int minPathSum(int[][] grid) {
        int m;
        if ((m = grid.length) == 0) {
            return -1;
        }
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < m; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int j = 1; j < n; j++) {
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.min(dp[i][j - 1], dp[i - 1][j]) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }

    //77. 组合
    public static List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        if (n < k) return res;

        LinkedList<Integer> select = new LinkedList<>();
        backtraceForCombine(n, k, 1, select, res);
        return res;
    }

    private static void backtraceForCombine(int n, int k, int start, LinkedList<Integer> select, List<List<Integer>> res) {
        if (select.size() == k) {
            res.add(new ArrayList<>(select));
            return;
        }
        //定义start，让选择不回头重新选择
        for (int i = start; i <= n; i++) {
            if (select.contains(i)) {
                continue;
            }
            select.add(i);
            backtraceForCombine(n, k, i + 1, select, res);
            select.removeLast();
        }
    }

    //78. 子集
    public static List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> whole = new ArrayList<>();
        List<Integer> select = new ArrayList<>();
        backForSubsets(nums, 0, select, whole);
        return whole;
    }

    private static void backForSubsets(int[] nums, int start, List<Integer> select, List<List<Integer>> whole) {
        whole.add(new ArrayList<>(select));
        for (int i = start; i < nums.length; i++) {
            select.add(nums[i]);
            backForSubsets(nums, i + 1, select, whole);
            select.remove(select.size() - 1);
        }
    }

    //90. 子集II
    public static List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> whole = new ArrayList<>();
        LinkedList<Integer> select = new LinkedList<>();
        Arrays.sort(nums);
        backtraceForSubsetsWithDup(nums, 0, select, whole);
        return whole;

    }

    private static void backtraceForSubsetsWithDup(int[] nums, int start, LinkedList<Integer> select, List<List<Integer>> res) {
        res.add(new ArrayList<>(select));
        for (int i = start; i < nums.length; i++) {
            //i>start，只对下次选择有影响，但不影响下次递归
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            select.add(nums[i]);
            backtraceForSubsetsWithDup(nums, i + 1, select, res);
            select.removeLast();
        }
    }

    public static void main(String[] args) {
//        System.out.println(threeSum(new int[]{-1, 1, 2, 11, 0, 1, -2}));
//        System.out.println(threeSum(new int[]{-1, 0, 1, 2, -1, -4}));
//
//        System.out.println(lengthOfLongestSubstring("dvdf"));
//        System.out.println(lengthOfLongestSubstring("pwwkew"));
//        System.out.println(lengthOfLongestSubstring("tmmzuxt"));
//        System.out.println(lengthOfLongestSubstring("tmmzzuxt"));
//        System.out.println(lengthOfLongestSubstring("tmmzuuzt"));
//        System.out.println(lengthOfLongestSubstring("tmmzutt"));
//        System.out.println(lengthOfLongestSubstring("tmmzuzt"));
//        System.out.println(lengthOfLongestSubstring("tmmzuzuzt"));
//        System.out.println(lengthOfLongestSubstring("abcabcbb"));
//        System.out.println(lengthOfLongestSubstring("bbbbb"));
//        System.out.println(lengthOfLongestSubstring("abcabbcad"));
//        System.out.println(lengthOfLongestSubstring("aa"));
//        System.out.println(lengthOfLongestSubstring("pwwwwkeeew"));
//        System.out.println(longestPalindromeV2("a"));

//        System.out.println(maxSubArray(new int[]{-2, 1, -3, 4, -1, 2, 1, -5, 4}));
//        System.out.println(numDecodings("2010"));
//        System.out.println(numDecodings("226"));
//        System.out.println(numDecodings("2360"));
//
//        System.out.println(permute(new int[]{1,2,3}));
//        System.out.println(generateParenthesis(0));
//        System.out.println(solveNQueens(8));

//        int[][] d = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
//        int x = 7, y = 7;
//        for (int i = 0; i < 4; i++) {reverseList
//            int m = x + d[i][0];
//            int n = y + d[i][1];
//            System.out.println(m + "——" + n);
//        }
//        ListNode z = new ListNode(0);
//        ListNode f = new ListNode(1);
//        ListNode s = new ListNode(2);
//        ListNode t = new ListNode(3);
//        ListNode four = new ListNode(4);
//        ListNode five = new ListNode(5);
//        ListNode six = new ListNode(6);
//        ListNode seven = new ListNode(7);
//        ListNode eight = new ListNode(8);
//        ListNode nine = new ListNode(9);
//
//        z.next = f;
//        f.next = s;
//        f.next = s;
//        s.next = t;
//        t.next = four;
//        four.next = five;
//        five.next = six;
//        six.next = seven;
//        seven.next = eight;
//        eight.next = nine;

//        ListNode result = removeNthFromEnd(f, 1);
//        ListNode result = reverseList(f);
//        ListNode result = reverseKGroup(f, 3);
//        ListNode result = rotateRight(null, 4);

//        int[][] one = new int[][]{{1}};
//        int[][] se = new int[][]{{1, 2}, {3, 4}};
//        int[][] th = new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
//        int[][] fo = new int[][]{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
//        int[][] fi = new int[][]{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}};
//        rotateMatrix(fi);

//        int[][] matrix = new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
//        int[][] matrix = new int[][]{{1, 2}};
//        System.out.println(spiralOrder(matrix));

//        54.
//        int[][] fi = new int[][]{{1, 3}, {2, 6}, {4, 5}, {7, 8}};
//        int[][] se = new int[][]{{1, 3}, {2, 6}, {8, 10}, {15, 18}};
//        int[][] one = new int[][]{{1, 3}};
//        merge(se);

//        59.
//        int[][] result = generateMatrix(3);

//        62.
//        System.out.println(uniquePaths(7, 3));

//        63.
//        System.out.println(uniquePathsWithObstacles(new int[][]{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}));

//        60.
//        System.out.println(getPermutation(3, 3));
//        System.out.println(getPermutation(4, 9));
//        System.out.println(getPermutation(1, 1));

//        21.
//        ListNode f = new ListNode(1);
//        ListNode s = new ListNode(2);
//        ListNode t = new ListNode(4);
//        f.next = s;
//        s.next = t;
//        ListNode f1 = new ListNode(1);
//        ListNode s1 = new ListNode(3);
//        ListNode t1 = new ListNode(4);
//        f1.next = s1;
//        s1.next = t1;
//        ListNode result = mergeTwoLists(f, f1);
//        System.out.println();

//        49.
//        System.out.println(groupAnagrams(new String[]{}));

//        50.
//        System.out.println(myPow(2.00000d, 10));
//        System.out.println(myPow(2.10000d, 3));
//        System.out.println(myPow(2.00000d, -2));

//        28.
//        System.out.println(strStr("abaclallb", "ll"));
//        System.out.println(strStr("aaaaa", "bba"));
//        System.out.println(strStr("hello", "ll"));
//        System.out.println(strStr("", "1"));
//        System.out.println(strStr("aaaaaab", "aab"));

//        64. 最小路径和
//        System.out.println(minPathSum(new int[][]{{1,3,1},{1,5,1},{4,2,1}}));

//        77. 组合
//        System.out.println(combine(4, 2));
//        System.out.println(combine(2, 2));

//        78.子集
//        System.out.println(subsets(new int[]{1, 2, 3}));
//        90. 子集II
        System.out.println(subsetsWithDup(new int[]{1, 2, 2, 3}));
        System.out.println(subsetsWithDup(new int[]{1, 2, 2}));
        System.out.println(subsetsWithDup(new int[]{}));
    }
}
