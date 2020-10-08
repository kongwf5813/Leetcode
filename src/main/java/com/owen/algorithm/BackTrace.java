package com.owen.algorithm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by OKONG on 2020/9/13.
 */
public class BackTrace {

    //17.电话号码的字母组合
    public static List<String> letterCombinations(String digits) {
        String[] choice = new String[]{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        List<String> result = new ArrayList<>();
        if (digits.length() == 0) {
            return result;
        }
        StringBuilder select = new StringBuilder();
        backtraceForLetterCombinations(choice, digits, 0, select, result);
        return result;
    }

    private static void backtraceForLetterCombinations(String[] choice, String digits, int start, StringBuilder select, List<String> result) {
        if (select.length() == digits.length()) {
            result.add(select.toString());
            return;
        }

        char[] chars = choice[digits.charAt(start) - '0'].toCharArray();
        for (int i = 0; i < chars.length; i++) {
            select.append(chars[i]);
            backtraceForLetterCombinations(choice, digits, start + 1, select, result);
            select.deleteCharAt(select.length() - 1);
        }
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

    //[39].组合总和
    public static List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (candidates.length == 0) return res;
        LinkedList<Integer> path = new LinkedList<>();
        backForCombinationSum(candidates, 0, target, res, path);
        return res;
    }

    private static void backForCombinationSum(int[] candidates, int start, int target, List<List<Integer>> res, LinkedList<Integer> path) {
        if (target == 0) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = start; i < candidates.length; i++) {
            if (candidates[i] > target) {
                continue;
            }
            path.add(candidates[i]);
            backForCombinationSum(candidates, i, target - candidates[i], res, path);
            path.removeLast();
        }
    }

    //[40].组合总和II
    public static List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (candidates.length == 0) return res;
        Arrays.sort(candidates);
        LinkedList<Integer> path = new LinkedList<>();
        backForcombinationSum2(candidates, 0, target, res, path);
        return res;
    }

    private static void backForcombinationSum2(int[] candidates, int start, int target, List<List<Integer>> res, LinkedList<Integer> path) {
        if (target == 0) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = start; i < candidates.length; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
            if (candidates[i] > target) {
                continue;
            }
            path.add(candidates[i]);
            backForcombinationSum2(candidates, i + 1, target - candidates[i], res, path);
            path.removeLast();
        }
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

    //[47].全排列II
    public static List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new LinkedList<>();
        if (nums.length == 0) return res;
        Arrays.sort(nums);
        LinkedList<Integer> tack = new LinkedList<>();
        boolean[] visited = new boolean[nums.length];
        backFroPermuteUnique(nums, visited, tack, res);
        return res;
    }

    private static void backFroPermuteUnique(int[] nums, boolean[] visited, LinkedList<Integer> select, List<List<Integer>> res) {
        if (select.size() == nums.length) {
            res.add(new ArrayList<>(select));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                if (visited[i - 1]) {
                    continue;
                }
            }
            if (visited[i]) {
                continue;
            }
            visited[i] = true;
            select.add(nums[i]);
            backFroPermuteUnique(nums, visited, select, res);
            select.removeLast();
            visited[i] = false;
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

    //[60].第k个排列(数学方法)
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

    //[79].单词搜索
    public static boolean exist(char[][] board, String word) {
        int m = board.length;
        int n = board[0].length;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (dfsForExist(board, i, j, 0, word, visited)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean dfsForExist(char[][] board, int x, int y, int index, String word, boolean[][] visited) {
        if (board[x][y] != word.charAt(index)) return false;
        if (index == word.length() - 1) return true;

        int[][] direct = new int[][]{{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
        int m = board.length;
        int n = board[0].length;
        visited[x][y] = true;
        //上下左右搜索
        for (int i = 0; i < 4; i++) {
            int newX = x + direct[i][0];
            int newY = y + direct[i][1];
            //边界检查，并且没有被访问过，则访问下一个方向
            if (0 <= newX && newX < m
                    && 0 <= newY && newY < n
                    && !visited[newX][newY]
                    && dfsForExist(board, newX, newY, index + 1, word, visited))
                return true;
        }

        visited[x][y] = false;
        return false;
    }

    //[89].格雷编码
    public static List<Integer> grayCode(int n) {
        boolean[] visited = new boolean[1 << n];
        LinkedList<Integer> select = new LinkedList<>();
        dfsForGrayCode(0, n, visited, select);
        return select;
    }

    private static boolean dfsForGrayCode(int cur, int n, boolean[] visited, LinkedList<Integer> select) {
        if (select.size() == 1 << n) {
            return true;
        }

        select.add(cur);
        visited[cur] = true;

        for (int i = 0; i < n; i++) {
            //选择下一个值，用异或生成一个合法值
            int next = cur ^ (1 << i);
            if (!visited[next] && dfsForGrayCode(next, n, visited, select)) {
                return true;
            }
        }
//        可以被注释掉，因为只要找到一个解就可以了
//        visited[cur] = false;
//        select.removeLast();
        return false;
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

    //[131]分割回文串
    public static List<List<String>> partition(String s) {
        int size = s.length();
        boolean[][] dp = new boolean[size][size];
        for (int i = 0; i < size; i++) {
            dp[i][i] = true;
        }
        for (int i = size - 2; i >= 0; i--) {
            for (int j = i + 1; j <= size - 1; j++) {
                dp[i][j] = (s.charAt(i) == s.charAt(j)) && (j - i < 3 || dp[i + 1][j - 1]);
            }
        }

        LinkedList<String> path = new LinkedList<>();
        List<List<String>> res = new ArrayList<>();
        backtraceForPartition(s, 0, path, res, dp);
        return res;
    }

    private static void backtraceForPartition(String s, int start, LinkedList<String> path, List<List<String>> res, boolean[][] dp) {
        if (start == s.length()) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = start; i < s.length(); i++) {
            if (!dp[start][i]) continue;
            path.add(s.substring(start, i + 1));
            backtraceForPartition(s, i + 1, path, res, dp);
            path.removeLast();
        }
    }

    //[216]组合总数III
    public static List<List<Integer>> combinationSum3(int k, int n) {
        LinkedList<Integer> select = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        if (n < 0) return res;
        backForCombinationSum3(select, res, 1, k, n);
        return res;
    }

    private static void backForCombinationSum3(LinkedList<Integer> select, List<List<Integer>> res, int start, int k, int target) {
        //k是多个元素，n是总数等于几，选择1～9
        if (k == select.size()) {
            if (target == 0) {
                res.add(new ArrayList<>(select));
            }
            return;
        }

        for (int i = start; i <= 9; i++) {
            select.add(i);
            backForCombinationSum3(select, res, i + 1, k, target - i);
            select.removeLast();
        }
    }

    public static void main(String[] args) {
//        17.电话号码的字母组合
//        letterCombinations("23");
//        letterCombinations("");
//
//        22. 括号生成
//        System.out.println(generateParenthesis(0));
//
//        [39].组合总和
//        combinationSum(new int[]{2, 3, 6, 7}, 7);

//        [40].组合总和II
//        combinationSum2(new int[]{10,1,2,7,6,1,5}, 8);
//        combinationSum2(new int[]{2,5,2,1,2}, 5);
//
//        [46]全排列
//        System.out.println(permute(new int[]{1, 2, 3}));
//
//        47. 全排列II
//        permuteUnique(new int[]{1, 1, 2});
//        permuteUnique(new int[]{3, 3, 0, 3});
//
//        51. N皇后
//        System.out.println(solveNQueens(8));
//
//        77. 组合
//        System.out.println(combine(4, 2));
//        System.out.println(combine(2, 2));
//
//        78. 子集
//        System.out.println(subsets(new int[]{1, 2, 3}));
//
//        79. 单词搜索
//        char[][] board = new char[][]{{'A', 'B', 'C', 'E'}, {'S', 'F', 'C', 'S'}, {'A', 'D', 'E', 'E'}};
//        System.out.println(exist(board, "ABCCED"));
//        System.out.println(exist(board, "ABCB"));
//        System.out.println(exist(board, "SEE"));
//        char[][] board2 = new char[][]{{'A'}};
//        System.out.println(exist(board2, "A"));
//
//        [89].格雷编码
//        System.out.println(grayCode(4));
//        System.out.println(grayCode(3));
//        System.out.println(grayCode(0));
//
//        90. 子集II
//        System.out.println(subsetsWithDup(new int[]{1, 2, 2, 3}));
//        System.out.println(subsetsWithDup(new int[]{1, 2, 2}));
//        System.out.println(subsetsWithDup(new int[]{}));
//
//        [131].分割回文串
//        System.out.println(partition("aba"));
//        System.out.println(partition("aab"));
//        System.out.println(partition("aabbaacbc"));

        System.out.println(combinationSum3(3, 9));
        System.out.println(combinationSum3(3, 7));
        System.out.println(combinationSum3(3, 1));
        System.out.println(combinationSum3(1, -1));

    }
}
