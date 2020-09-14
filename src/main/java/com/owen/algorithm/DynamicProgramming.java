package com.owen.algorithm;

/**
 * Created by OKONG on 2020/9/13.
 */
public class DynamicProgramming {

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

    //[198].打家劫舍
    public static int rob(int[] nums) {
        int size = nums.length;
        if (size == 0) return 0;
        if (size == 1) return nums[0];
        int[] dp = new int[size + 1];
        //the maximum money robbed from i to n
        //base case
        dp[size - 1] = nums[size - 1];
        for (int i = nums.length - 2; i >= 0; i--) {
            //选择打劫i是nums[i] + dp[i + 2]，不打劫i是dp[i + 1]
            dp[i] = Math.max(nums[i] + dp[i + 2], dp[i + 1]);
        }
        return dp[0];
    }

    public static void main(String[] args) {
//        [5]最长回文子串
//        System.out.println(longestPalindrome("a"));
//        System.out.println(longestPalindromeV2("a"));
//
//        [53]最大子序列和
//        System.out.println(maxSubArray(new int[]{-2, 1, -3, 4, -1, 2, 1, -5, 4}));
//
//        62.不同路径 动态规划
//        System.out.println(uniquePaths(7, 3));
//
//        63.不同路径II
//        System.out.println(uniquePathsWithObstacles(new int[][]{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}));
//
//        64.最小路径和
//        System.out.println(minPathSum(new int[][]{{1, 3, 1}, {1, 5, 1}, {4, 2, 1}}));
//
//        [91]解码方法
//        System.out.println(numDecodings("2010"));
//        System.out.println(numDecodings("226"));
//        System.out.println(numDecodings("2360"));
//
//        [96].不同的二叉搜索树
//        System.out.println(numTrees(3));
//        System.out.println(numTrees(5));
//        System.out.println(numTrees(2));
//        System.out.println(numTrees(1));

//        121.买卖股票的最佳时机
//        System.out.println(maxProfit(new int[]{2, 1, 4, 9}));
//        System.out.println(maxProfit(new int[]{7, 4, 3, 1}));
//        System.out.println(maxProfit(new int[]{7, 1, 5, 3, 6, 4}));
//
//        198.打家劫舍
//        System.out.println(rob(new int[]{1, 2, 3, 1}));
//        System.out.println(rob(new int[]{2, 7, 9, 3, 1}));
//        System.out.println(rob(new int[]{2, 2, 1}));
//        System.out.println(rob(new int[]{2, 7, 9, 3, 1}));
    }

}
