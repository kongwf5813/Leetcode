package com.owen.algorithm.v3;

import com.amazonaws.services.dynamodbv2.xspec.S;

import it.unimi.dsi.fastutil.Hash;

import java.util.HashSet;
import java.util.Set;

public class Test {

    int numDistinctIslands(int[][] grid) {

        Set<String> set = new HashSet<>();
        for(int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    StringBuilder sb = new StringBuilder();
                    dfs(grid, i, j, i, j, sb);
                    set.add(sb.toString());
                }
            }
        }
        return set.size();
    }

    void dfs(int[][] grid, int x, int y, int originalX, int originalY, StringBuilder sb) {
        if (x < 0 || y < 0 || x >= grid.length || y >= grid[0].length || grid[x][y] == 0) return;

        grid[x][y] = 0;
        sb.append(originalX - x);
        sb.append(originalY - y);

        dfs(grid, x -1, y, originalX, originalY, sb);
        dfs(grid, x +1, y, originalX, originalY, sb);
        dfs(grid, x, y-1, originalX, originalY, sb);
        dfs(grid, x, y+1, originalX, originalY, sb);
    }
}