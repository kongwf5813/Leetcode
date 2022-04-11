package com.owen.algorithm.v3;


import com.owen.algorithm.v3.AllOfThem.*;

import java.util.Arrays;


public class Test {


    public static void main(String[] args) {
        System.out.println(numSquares(7));
    }

    public static int numSquares(int n) {
        if (n <= 0) return 0;
        int[] dp = new int[n + 1];
        Arrays.fill(dp, 0x3f3f3f3f);
        dp[0] = 0;
        //物品是完全平方数
        for (int i = 1; i * i <= n; i++) {
            //背包是n
            for (int j = i * i; j <= n; j++) {
                dp[j] = Math.min(dp[j - i * i] + 1, dp[j]);
            }
        }
        return dp[n];
    }
}