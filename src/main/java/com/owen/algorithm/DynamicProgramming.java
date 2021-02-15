package com.owen.algorithm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by OKONG on 2020/9/13.
 */
public class DynamicProgramming {

    //[5].最长回文子串
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

    //[53].最大子序列和
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

    //[62].不同路径 动态规划
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

    //[63].不同路径II
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

    //[64].最小路径和
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

    //[72].编辑距离
    public static int minDistance(String word1, String word2) {

        int m = word1.length();
        int n = word2.length();
        //最小编辑距离
        int[][] dp = new int[m + 1][n + 1];

        for (int i = 1; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = j;
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1), dp[i - 1][j - 1] + 1);
                }
            }
        }
        return dp[m][n];
    }

    //[91].解码方法
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

    //[96].不同的二叉搜索树(卡特兰数 fn = f0*fn-1 + f1*fn-2+ ...+ fn-1*f0)
    public static int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i < n + 1; i++) {
            for (int j = 0; j < i; j++) {
                dp[i] += dp[j] * dp[i - j - 1];
            }
        }
        return dp[n];
    }

    //[120]三角形最小路径和
    public static int minimumTotal(List<List<Integer>> triangle) {
        int m = triangle.size();
        int[] dp = new int[m];
        dp[0] = triangle.get(0).get(0);
        int pre = 0, cur;
        for (int i = 1; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                cur = dp[j];
                if (j == 0) {
                    dp[j] = cur + triangle.get(i).get(j);
                } else if (j == i) {
                    dp[j] = pre + triangle.get(i).get(j);
                } else {
                    dp[j] = Math.min(pre, cur) + triangle.get(i).get(j);
                }
                pre = cur;
            }
        }
        int res = dp[0];
        for (int i = 1; i < m; i++) {
            res = Math.min(res, dp[i]);
        }
        return res;
    }

    //[121].买卖股票的最佳时机
    public static int maxProfit(int[] prices) {
        int n = prices.length;
        if (n == 0) return 0;
        int[][] dp = new int[n][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for (int i = 1; i < n; i++) {
            //卖的最大利润(今天的利润，与昨天的利润哪个最多)
            dp[i][0] = Math.max(dp[i - 1][1] + prices[i], dp[i - 1][0]);
            //买的最大利润(上一天与今天亏的，哪个亏的更多)
            dp[i][1] = Math.max(dp[i - 1][1], -prices[i]);
        }
        return dp[n - 1][0];
    }

    //[122].买卖股票的最佳时机II
    public static int maxProfit2(int[] prices) {
        // 第i天最多交易k次没有股票 dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
        // 第i天最多交易k次持有股票 dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
        int n = prices.length;
        if (n == 0) return 0;
        int[][] dp = new int[n][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for (int i = 1; i < n; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }
        return dp[n - 1][0];
    }

    //[123].买卖股票的最佳时机III
    public static int maxProfit3(int[] prices) {
        int dp_i10 = 0, dp_i11 = Integer.MIN_VALUE;
        int dp_i20 = 0, dp_i21 = Integer.MIN_VALUE;

        for (int num : prices) {
            dp_i20 = Math.max(dp_i20, dp_i21 + num);
            dp_i21 = Math.max(dp_i21, dp_i10 - num);

            dp_i10 = Math.max(dp_i10, dp_i11 + num);
            dp_i11 = Math.max(dp_i11, -num);
        }
        return dp_i20;
        /*
        写法二
        int n = prices.length;
        int maxK = 2;
        int[][][] dp = new int[n][maxK + 1][2];
        dp[0][1][0] = 0;
        dp[0][1][1] = -prices[0];
        dp[0][2][0] = 0;
        dp[0][2][1] = -prices[0];
        for (int i = 1; i< n;i++) {
            for (int k = 1; k <= maxK; k++) {
                dp[i][k][0] = Math.max(dp[i-1][k][0], dp[i-1][k][1] + prices[i]);
                dp[i][k][1] = Math.max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i]);
            }
        }
        return dp[n-1][maxK][0];
        */
    }

    //[188].买卖股票的最佳时机IV
    public static int maxProfit4(int k, int[] prices) {
        int n = prices.length;
        if (n == 0) return 0;
        if (k > n / 2) {
            int[][] dp1 = new int[n][2];
            dp1[0][0] = 0;
            dp1[0][1] = -prices[0];
            for (int i = 1; i < n; i++) {
                dp1[i][0] = Math.max(dp1[i - 1][0], dp1[i - 1][1] + prices[i]);
                dp1[i][1] = Math.max(dp1[i - 1][1], dp1[i - 1][0] - prices[i]);
            }
            return dp1[n - 1][0];
        }

        int[][][] dp = new int[n][k + 1][2];
        for (int i = 1; i <= k; i++) {
            dp[0][i][0] = 0;
            dp[0][i][1] = -prices[0];
        }
        for (int i = 1; i < n; i++) {
            for (int j = 1; j <= k; j++) {
                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
                dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i]);
            }
        }
        return dp[n - 1][k][0];
    }

    //[139].单词拆分
    public static boolean wordBreak(String s, List<String> wordDict) {
        //dp[i] 前i个是否能拆分
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = i - 1; j >= 0; j--) {
                dp[i] = dp[j] & wordDict.contains(s.substring(j, i));
                if (dp[i]) break;
            }
        }
        return dp[s.length()];
    }

    //[152].乘积最大子数组
    public static int maxProduct(int[] nums) {
        int size = nums.length;
        //dp[i][0] 表示最大值， dp[i][1]表示最小值
        int[][] dp = new int[size][2];
        dp[0][0] = nums[0];
        dp[0][1] = nums[0];
        int max = nums[0];
        for (int i = 1; i < size; i++) {
            if (nums[i] > 0) {
                dp[i][0] = Math.max(dp[i - 1][0] * nums[i], nums[i]);
                dp[i][1] = Math.min(dp[i - 1][1] * nums[i], nums[i]);
            } else {
                //上一次的最小值可能是负数 * 负数 = 最大值
                dp[i][0] = Math.max(dp[i - 1][1] * nums[i], nums[i]);
                //上一次的最大值可能是正数 * 负数 = 最小值
                dp[i][1] = Math.min(dp[i - 1][0] * nums[i], nums[i]);
            }
            max = Math.max(dp[i][0], max);
        }
        return max;
    }

    //[198].打家劫舍
    public static int rob(int[] nums) {
        int size = nums.length;
        if (size == 0) return 0;
        if (size == 1) return nums[0];
        int[] dp = new int[size + 1];
        //the maximum money robbed from i to n
        //base case
        dp[size - 1] = nums[size - 1];
        //自底向上
        for (int i = nums.length - 2; i >= 0; i--) {
            //选择打劫i是nums[i] + dp[i + 2]，不打劫i是dp[i + 1]
            dp[i] = Math.max(nums[i] + dp[i + 2], dp[i + 1]);
        }
        return dp[0];
    }

    //[213].打家劫舍II
    public static int rob2(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        int res = Math.max(rob2(nums, 0, nums.length - 2), rob2(nums, 1, nums.length - 1));
        return res;
    }

    private static int rob2(int[] nums, int start, int end) {
        int dp_i_2 = 0, dp_i_1 = 0, dp_i = 0;
        for (int i = end; i >= start; i--) {
            dp_i = Math.max(dp_i_2 + nums[i], dp_i_1);
            dp_i_2 = dp_i_1;
            dp_i_1 = dp_i;
        }
        return dp_i;
    }

    //[221].最大正方形
    public static int maximalSquare(char[][] matrix) {
        int row = matrix.length;
        if (row == 0) return 0;

        int col = matrix[0].length;
        //dp[i][j]为(0,0) -> (i,j)的最大边长
        int[][] dp = new int[row][col];
        int maxSide = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(dp[i - 1][j], Math.min(dp[i - 1][j - 1], dp[i][j - 1])) + 1;
                    }
                    maxSide = Math.max(dp[i][j], maxSide);
                }
            }
        }
        return maxSide * maxSide;
    }

    //[279].完全平方数
    public static int numSquares(int n) {
        if (n <= 0) return 0;
        int[] dp = new int[n + 1];
        dp[0] = 0;
        //自顶向下
        for (int i = 1; i <= n; i++) {
            dp[i] = Integer.MAX_VALUE;
            for (int j = 1; j * j <= i; j++) {
                dp[i] = Math.min(dp[i - j * j] + 1, dp[i]);
            }
        }
        return dp[n];
    }

    //[300]最长上升子序列
    public static int lengthOfLIS(int[] nums) {
        int size = nums.length;
        if (size == 0) return 0;
        int[] dp = new int[size];
        int length = 1;
        for (int i = 0; i < size; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            length = Math.max(length, dp[i]);
        }
        return length;
    }

    //[304].二位区域和检索-矩阵不可变
    static class NumMatrix {
        int[][] dp;

        public NumMatrix(int[][] matrix) {
            if (matrix.length > 0 && matrix[0].length > 0) {
                dp = new int[matrix.length][matrix[0].length];
                for (int i = 0; i < matrix.length; i++) {
                    for (int j = 0; j < matrix[0].length; j++) {
                        if (i == 0 && j == 0) {
                            dp[i][j] = matrix[i][j];
                        } else if (i == 0) {
                            dp[i][j] = dp[i][j - 1] + matrix[i][j];
                        } else if (j == 0) {
                            dp[i][j] = dp[i - 1][j] + matrix[i][j];
                        } else {
                            dp[i][j] = dp[i - 1][j] + dp[i][j - 1] - dp[i - 1][j - 1] + matrix[i][j];
                        }
                    }
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            if (dp == null || dp.length == 0) return 0;
            int left = col1 != 0 ? dp[row2][col1 - 1] : 0;
            int top = row1 != 0 ? dp[row1 - 1][col2] : 0;
            int leftTop = col1 != 0 && row1 != 0 ? dp[row1 - 1][col1 - 1] : 0;
            return dp[row2][col2] - left - top + leftTop;
        }
    }

    //[309].最佳买卖股票时机含冷冻期
    public static int maxProfit4(int[] prices) {
        int n = prices.length;
        if (n <= 1) return 0;
        int[][] dp = new int[n][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        dp[1][0] = Math.max(dp[0][0], dp[0][1] + prices[1]);
        dp[1][1] = Math.max(dp[0][1], -prices[1]);
        for (int i = 2; i < n; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 2][0] - prices[i]);
        }
        return dp[n - 1][0];
    }

    //[313].超级丑数
    public static int nthSuperUglyNumber(int n, int[] primes) {
        int k = primes.length;
        int[] pointer = new int[k];

        int[] dp = new int[n];
        dp[0] = 1;

        for (int i = 1; i < n; i++) {
            int min = Integer.MAX_VALUE;
            //找到min值
            for (int j = 0; j < k; j++) {
                int val = dp[pointer[j]] * primes[j];
                if (val < min) {
                    min = val;
                }
            }

            dp[i] = min;

            //更新指针
            for (int j = 0; j < k; j++) {
                //可能不止一个
                if (min == dp[pointer[j]] * primes[j]) {
                    pointer[j]++;
                }
            }
        }
        return dp[n - 1];
    }

    //[322].兑换零钱
    public static int coinChange(int[] coins, int amount) {
        //总金额为i最少需要多少硬币
        int[] dp = new int[amount + 1];
        dp[0] = 0;
        for (int i = 1; i < dp.length; i++) {
            dp[i] = amount + 1;
            for (int coin : coins) {
                if (i < coin) continue;
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }

    //[337].打家劫舍III
    private static Map<Tree.TreeNode, Integer> robMap = new HashMap<>();

    public static int rob(Tree.TreeNode root) {
        if (root == null) return 0;
        if (robMap.containsKey(root)) return robMap.get(root);
        int rob_it = root.val +
                (root.left != null ? rob(root.left.left) + rob(root.left.right) : 0) +
                (root.right != null ? rob(root.right.left) + rob(root.right.right) : 0);

        int rob_not = rob(root.left) + rob(root.right);
        int res = Math.max(rob_it, rob_not);
        robMap.put(root, res);
        return res;
    }

    //[338].比特位计数
    public static int[] countBits(int num) {
        //000 001 010 011 100 101 110 111 1000 1001  1010  1011  1100   1101  1110 1111  10000
        //0   1    2   3   4   5   6   7   8    9     10    11    12     13    14   15    16
        //0   1    1   2   1   2   2   3   1    2     2     3      2      3    3    4     1
        int[] dp = new int[num + 1];
        dp[0] = 0;
        int power = 1;
        for (int i = 1; i <= num; i++) {
            if (i == 1 << power) {
                dp[i] = 1;
                power++;
            } else {
                int recent = (1 << (power - 1)) - 1;
                dp[i] = dp[recent & i] + 1;
            }
        }
        return dp;
    }

    //[343].整数拆分
    public static int integerBreak(int n) {
        //1  2  3   4    5
        //0  1  2  2*2   4
        int[] dp = new int[n + 1];
        //dp[j] * (i-j)  (i-j) * j
        for (int i = 1; i <= n; i++) {
            //j作为被拆的数，剩下的数是j-i
            for (int j = 1; j < i; j++) {
                dp[i] = Math.max(dp[i], Math.max(dp[j] * (i - j), j * (i - j)));
            }
        }
        return dp[n];
    }

    //[354].俄罗斯套娃信封问题
    public static int maxEnvelopes(int[][] envelopes) {
        //宽度增序，高度降序
        Arrays.sort(envelopes, (a,b)-> a[0] == b[0] ? b[1] - a[1] : a[0] - b[0]);

        //求高度的递增子序列
        int[] nums = new int[envelopes.length];
        for (int i = 0; i< envelopes.length; i++) {
            nums[i] = envelopes[i][1];
        }
        return lengthOfLIS(nums);
    }
    //[357].计算各个位数不同的数字个数
    public static int countNumbersWithUniqueDigits(int n) {
        if (n == 0) return 1;
        if (n == 1) return 10;
        int[] dp = new int[n + 1];
        //dp[i]: 数字为i位，组合成不重复的数字的个数
        // dp[1] = 9
        // dp[2] = 9*9
        // dp[3] = 9*9*8
        dp[0] = 1;
        dp[1] = 9;
        int res = 10;
        for (int i = 2; i <= Math.min(n, 10); i++) {
            dp[i] = dp[i - 1] * (10 - i + 1);
            res += dp[i];
        }
        return res;
    }

    //[368].最大整除子集
    public static List<Integer> largestDivisibleSubset(int[] nums) {
        List<Integer> res = new ArrayList<>();
        int size = nums.length;
        if (size == 0) return res;

        Arrays.sort(nums);
        int maxNum = 1;
        int maxIndex = 0;
        //前i个数的最大子集个数是多少
        int[] dp = new int[size];
        for (int i = 0; i < size; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] % nums[j] == 0) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            if (dp[i] > maxNum) {
                maxNum = dp[i];
                maxIndex = i;
            }
        }

        for (int i = maxIndex; i >= 0; i--) {
            //最大子集个数 == 当前位置的最大子集个数 && 最大值能被当前整数整除
            if (nums[maxIndex] % nums[i] == 0 && dp[i] == maxNum) {
                res.add(0, nums[i]);
                //最大子集个数减一
                maxNum--;
                //替换到另一个
                maxIndex = i;
            }
        }
        return res;
    }

    //[372].猜数字大小 II
    public static int getMoneyAmount(int n) {
        int[][] dp = new int[n + 1][n + 1];
        for (int i = n - 1; i >= 1; i--) {
            for (int j = i; j <= n; j++) {
                if (i == j) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = Integer.MAX_VALUE;
                    for (int x = i; x < j; x++) {
                        dp[i][j] = Math.min(Math.max(dp[i][x - 1], dp[x + 1][j]) + x, dp[i][j]);
                    }
                }
            }
        }
        return dp[1][n];
    }

    //[376].摆动序列
    public static int wiggleMaxLength(int[] nums) {
        if (nums.length < 2) return nums.length;
        //i为结尾的上升子序列的最大长度
        int[] up = new int[nums.length];
        //i为结尾的下降子序列的最大长度
        int[] down = new int[nums.length];

        down[0] = 1;
        up[0] = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > nums[i - 1]) {
                //上升, 以i-1为结尾的下降子序列的最大长度+1
                up[i] = down[i - 1] + 1;
                down[i] = down[i - 1];
            } else if (nums[i] < nums[i - 1]) {
                //下降, 以i-1为结尾的上升子序列的最大长度+1
                down[i] = up[i - 1] + 1;
                up[i] = up[i - 1];
            } else {
                down[i] = down[i - 1];
                up[i] = up[i - 1];
            }
        }
        return Math.max(up[nums.length - 1], down[nums.length - 1]);
    }

    //[377].组合总和IV
    public static int combinationSum4(int[] nums, int target) {

        //和为i的组合总数
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i <= target; i++) {
            for (int num : nums) {
                //选择num之后，可以有多少种
                if (i - num >= 0) {
                    dp[i] += dp[i - num];
                }
            }
        }
        return dp[target];
    }

    //[396].旋转函数
    public static int maxRotateFunction(int[] A) {
        //F(n) - F(n-1) = Sum - size * A[size -n]
        int dp0 = 0;
        int sum = 0;
        int size = A.length;
        for (int i = 0; i < size; i++) {
            dp0 += i * A[i];
            sum += A[i];
        }
        int dp1;
        int max = dp0;
        for (int i = 1; i < size; i++) {
            dp1 = dp0 + sum - size * A[size - i];
            max = dp1 > max ? dp1 : max;
            dp0 = dp1;
        }
        return max;
    }

    //[413].等差数列划分
    public static int numberOfArithmeticSlices(int[] A) {
        //dp[i] 0~i 等差数列新增的个数
        // 1 2 3 4 5
        //dp[2] = 1
        //dp[3] = 2
        //dp[4] = 3
        //sum  = 6个

        int size = A.length;
        int[] dp = new int[size];
        int sum = 0;
        for (int i = 2; i < size; i++) {
            if (A[i] - A[i - 1] == A[i - 1] - A[i - 2]) {
                dp[i] = dp[i - 1] + 1;
                sum += dp[i];
            }
        }
        return sum;
    }

    //[416].分割等和子集
    public static boolean canPartition(int[] nums) {
        if (nums.length < 2) return false;
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        if (sum % 2 != 0) return false;
        sum = sum / 2;
        //dp[i][j] [0,i]范围内选取若干个，是否存在被选取的正整数的和等于j
        boolean[][] dp = new boolean[nums.length][sum + 1];
        //base case
        for (int i = 0; i < nums.length; i++) {
            dp[i][0] = true;
        }
        for (int i = 1; i < nums.length; i++) {
            for (int j = 1; j <= sum; j++) {
                if (j < nums[i]) {
                    //第i个不放入
                    dp[i][j] = dp[i - 1][j];
                } else {
                    //取决于第i个不放入 和 第i个放入两种情况
                    dp[i][j] = dp[i - 1][j] | dp[i - 1][j - nums[i]];
                }
            }
        }
        return dp[nums.length - 1][sum];
    }

    //[474].一和零
    public static int findMaxForm(String[] strs, int m, int n) {
        int[][] dp = new int[m + 1][n + 1];
        for (String str : strs) {
            int zeros = 0, ones = 0;
            for (char ch : str.toCharArray()) {
                if (ch == '0') zeros++;
                else ones++;
            }
            for (int i = m; i >= zeros; i--) {
                for (int j = n; j >= ones; j--) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - zeros][j - ones] + 1);
                }
            }
        }
        return dp[m][n];
    }

    //[486].预测赢家
    public static boolean PredictTheWinner(int[] nums) {
        if (nums == null || nums.length == 0) return false;
        int n = nums.length;
        //dp[i][j] 从i到j先手玩家比后手玩家多的最大分数
        int[][] dp = new int[n][n];

        //一个数字选择那么就是nums[i]
        for (int i = 0; i < n; i++) {
            dp[i][i] = nums[i];
        }

        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                dp[i][j] = Math.max(nums[i] + (-dp[i + 1][j]), nums[j] + (-dp[i][j - 1]));
            }
        }
        return dp[0][n - 1] >= 0;
    }

    //[509].斐波那契数
    public static int fib(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;
        int dp_i_1 = 1, dp_i_2 = 0, dp_i = 0;
        for (int i = 1; i < n; i++) {
            dp_i = dp_i_1 + dp_i_2;
            dp_i_2 = dp_i_1;
            dp_i_1 = dp_i;
        }
        return dp_i;
    }

    //[516].最长回文子序列
    public static int longestPalindromeSubseq(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];
        for (int i = 0; i < n; i++) {
            dp[i][i] = 1;
        }

        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }

    //[518].兑换零钱II
    public static int change(int amount, int[] coins) {
        int size = coins.length;
        if (size == 0) {
            if (amount == 0) {
                return 1;
            }
            return 0;
        }

        //前i种硬币中，凑成j的总数
        int[][] dp = new int[size][amount + 1];
        for (int i = 0; i < size; i++) {
            dp[i][0] = 1;
        }

        for (int j = 1; j <= amount; j++) {
            dp[0][j] = j % coins[0] == 0 ? 1 : 0;
        }

        for (int i = 1; i < size; i++) {
            for (int j = 1; j <= amount; j++) {
                if (j >= coins[i]) {
                    dp[i][j] = dp[i][j - coins[i]] + dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[size - 1][amount];
    }

    //[581].两个字符串的删除操作
    public static int minDistance2(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int lcs = longestCommonSubsequence(word1, word2);
        return m - lcs + n - lcs;
    }

    //[712].两个字符串的最小ASCII删除和
    public static int minimumDeleteSum(String s1, String s2) {
        int m = s1.length(), n = s2.length();

        int sum = 0;
        for (int i = 0; i < m; i++) sum += s1.charAt(i);
        for (int i = 0; i < n; i++) sum += s2.charAt(i);
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + (int) (s1.charAt(i - 1));
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return sum - (dp[m][n] * 2);
    }

    //[714].买卖股票的最佳时机含手续费
    public static int maxProfit(int[] prices, int fee) {
        int n = prices.length;
        int[][] dp = new int[n][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for (int i = 1; i < n; i++) {
            //没有股票
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee);
            //有股票
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }
        return dp[n - 1][0];
    }

    //[1143].最长公共子序列
    public static int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        if (m == 0 || n == 0) return 0;
        //dp[i][j]定义, 从text1[..i] text2[..j]的公共子序列最大长度
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }

    public static void main(String[] args) {
//        [5].最长回文子串
//        System.out.println(longestPalindrome("a"));
//        System.out.println(longestPalindromeV2("a"));
//
//        [53].最大子序列和
//        System.out.println(maxSubArray(new int[]{-2, 1, -3, 4, -1, 2, 1, -5, 4}));
//
//        [62].不同路径 动态规划
//        System.out.println(uniquePaths(7, 3));
//
//        [63].不同路径II
//        System.out.println(uniquePathsWithObstacles(new int[][]{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}));
//
//        [64].最小路径和
//        System.out.println(minPathSum(new int[][]{{1, 3, 1}, {1, 5, 1}, {4, 2, 1}}));
//
//        [72].编辑距离
//        System.out.println(minDistance("", "1"));
//
//        [91].解码方法
//        System.out.println(numDecodings("2010"));
//        System.out.println(numDecodings("226"));
//        System.out.println(numDecodings("2360"));
//
//        [96].不同的二叉搜索树
//        System.out.println(numTrees(3));
//        System.out.println(numTrees(5));
//        System.out.println(numTrees(2));
//        System.out.println(numTrees(1));
//
//        [120].三角形最小路径和
//        List<List<Integer>> triangle = new ArrayList<>();
//        triangle.add(Arrays.asList(2));
//        triangle.add(Arrays.asList(3, 4));
//        triangle.add(Arrays.asList(6, 5, 7));
//        triangle.add(Arrays.asList(4, 1, 8, 3));
//        System.out.println(minimumTotal(triangle));
//
//        [121].买卖股票的最佳时机
//        System.out.println(maxProfit(new int[]{2, 1, 4, 9}));
//        System.out.println(maxProfit(new int[]{7, 4, 3, 1}));
//        System.out.println(maxProfit(new int[]{7, 1, 5, 3, 6, 4}));
//
//        [122].买卖股票的最佳时机II
//        System.out.println(maxProfit2(new int[]{7,1,5,3,6,4}));
//        System.out.println(maxProfit2(new int[]{1,2,3,4,5}));
//        System.out.println(maxProfit2(new int[]{7,6,4,3,1}));
//
//        [123].买卖股票的最佳时机III
//        System.out.println(maxProfit3(new int[]{1, 2, 3, 4, 5}));
//
//        [139].单词拆分
//        System.out.println(wordBreak("catsandog", Arrays.asList("cats", "dog", "sand", "and", "cat")));
//
//        [152].乘积最大子数组
//        System.out.println(maxProduct(new int[]{2,3,-2,4}));
//        System.out.println(maxProduct(new int[]{2,3,0,4}));
//
//        [188].买卖股票的最佳时机IV
//        System.out.println(maxProfit4(2, new int[]{3,2,6,5,0,3}));
//        System.out.println(maxProfit4(1, new int[]{1,2}));
//        System.out.println(maxProfit4(2, new int[]{2,4,1}));
//        System.out.println(maxProfit4(4, new int[]{1,2,4,2,5,7,2,4,9,0}));
//        System.out.println(maxProfit4(4, new int[]{5,7,2,7,3,3,5,3,0}));
//
//        [198].打家劫舍
//        System.out.println(rob(new int[]{1, 2, 3, 1}));
//        System.out.println(rob(new int[]{2, 7, 9, 3, 1}));
//        System.out.println(rob(new int[]{2, 2, 1}));
//        System.out.println(rob(new int[]{2, 7, 9, 3, 1}));
//
//        [213].打家劫舍II
//        System.out.println(rob2(new int[]{1, 2, 3, 4}));
//        System.out.println(rob2(new int[]{2, 3, 2}));
//
//        [221].最大正方形
//        char[][] area = new char[][]{{'1', '0', '1', '0', '0'}, {'1', '0', '1', '1', '1'}, {'1', '1', '1', '1', '1'}, {'1', '0', '0', '1', '0'}};
//
//        [279].完全平方数
//        System.out.println(numSquares(13));
//        System.out.println(numSquares(12));
//        System.out.println(numSquares(16));
//        System.out.println(numSquares(1));
//        System.out.println(maximalSquare(area));
//
//        [304].二位区域和检索-矩阵不可变
//        NumMatrix numMatrix = new NumMatrix(new int[][]{{3, 0, 1, 4, 2}, {5, 6, 3, 2, 1}, {1, 2, 0, 1, 5}, {4, 1, 0, 1, 7}, {1, 0, 3, 0, 5}});
//        System.out.println(numMatrix.sumRegion(2, 1, 4, 3));// -> 8
//        System.out.println(numMatrix.sumRegion(1, 1, 2, 2));// -> 11
//        System.out.println(numMatrix.sumRegion(1, 2, 2, 4));//-> 12
//
//        [309].最佳买卖股票时机含冷冻期
//        System.out.println(maxProfit(new int[] {1,2,3,0,2}));
//
//        [313].超级丑数
//        System.out.println(nthSuperUglyNumber(12, new int[]{2, 7, 13, 19}));
//
//        [322].兑换零钱
//        System.out.println(coinChange(new int[] {2}, 3));
//        System.out.println(coinChange(new int[] {1}, 0));
//        System.out.println(coinChange(new int[] {1,2,5}, 11));
//
//        [343].整数拆分
//        System.out.println(integerBreak(10));
//

        System.out.println(maxEnvelopes(new int[][]{{5,4},{6,4},{6,7},{2,3}}));
//        [368].最大整除子集
//        System.out.println(largestDivisibleSubset(new int[]{1, 2, 3, 4, 8}));
//        System.out.println(largestDivisibleSubset(new int[]{1}));
//
//        [375].猜数字大小 II
//        System.out.println(getMoneyAmount(8));
//        System.out.println(getMoneyAmount(1));
//
//        [376].摆动序列
//        System.out.println(wiggleMaxLength(new int[] {1,7,4,9,2,5}));
//        System.out.println(wiggleMaxLength(new int[] {1,2,1,0,-1,1}));
//
//        [377].组合总和IV
//        System.out.println(combinationSum4(new int[]{1, 2, 3}, 4));
//
//        [413].等差数列划分
//        System.out.println(numberOfArithmeticSlices(new int[] {1,2,3, 4}));
//        System.out.println(numberOfArithmeticSlices(new int[] {1,2,3,4,5}));
//
//        [416].分割等和子集
//        System.out.println(canPartition(new int[]{1, 5, 11, 5}));
//
//        [474].一和零
//        System.out.println(findMaxForm(new String[]{"10", "0001", "111001", "1", "0"}, 5, 3));
//        System.out.println(findMaxForm(new String[]{"10", "0", "1"}, 1, 1));
//
//        [486].预测赢家
//        System.out.println(PredictTheWinner(new int[]{1, 5, 233, 7}));
//        System.out.println(PredictTheWinner(new int[]{1, 5, 2}));
//
//        [509].斐波那契数
//        System.out.println(fib(0));
//        System.out.println(fib(1));
//        System.out.println(fib(2));
//        System.out.println(fib(3));
//        System.out.println(fib(4));
//
//        [516].最长回文子序列
//        System.out.println(longestPalindromeSubseq("bb"));
//
//        [518].兑换零钱II
//        System.out.println(change(5, new int[]{1, 2, 5}));
//        System.out.println(change(3, new int[]{2}));
//        System.out.println(change(0, new int[]{}));
//        System.out.println(change(7, new int[]{}));
//
//        [581].两个字符串的删除操作
//        System.out.println(minDistance2("sea", "eat"));
//
//        [712].两个字符串的最小ASCII删除和
//        System.out.println(minimumDeleteSum("delete", "leet"));
//
//        [714].买卖股票的最佳时机含手续费
//        System.out.println(maxProfit(new int[]{1, 3, 2, 8, 4, 9}, 2));
//
//        [1143].最长公共子序列
//        System.out.println(longestCommonSubsequence("abcde", "ace"));
    }
}