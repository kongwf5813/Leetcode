package com.owen.algorithm;

import java.util.*;
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

    //[306].累加数
    public static boolean isAdditiveNumber(String num) {
        return dfsForAdditiveNumber(0, 0, 0, num, 0);
    }

    private static boolean dfsForAdditiveNumber(long sum, long pre, int start, String num, int k) {
        if (start == num.length()) {
            //遍历完，并且要保证有三个数操作
            return k > 2;
        }
        for (int i = start; i < num.length(); i++) {
            //选择一个值，进行判断
            long cur = selectNumber(num, start, i);
            if (cur == -1) {
                continue;
            }

            //初始阶段是前两次必然不等, 但是得选择110
            //剪枝，不相等，说明不合法
            if (k >= 2 && sum != cur) {
                continue;
            }

            //合法值，进行选择
            if (dfsForAdditiveNumber(pre + cur, cur, i + 1, num, k + 1)) {
                return true;
            }
        }
        //全部遍历完也没发现合法的
        return false;
    }

    private static long selectNumber(String num, int l, int r) {
        if (l < r && num.charAt(l) == '0') {
            return -1;
        }
        long res = 0;
        while (l <= r) {
            res = res * 10 + num.charAt(l++) - '0';
        }
        return res;
    }

    //[332].重新安排行程
    public static List<String> findItinerary(List<List<String>> tickets) {
        List<String> res = new LinkedList<>();
        if (tickets == null || tickets.size() == 0) {
            return res;
        }
        Map<String, List<String>> graph = new HashMap<>();
        for (List<String> ticket : tickets) {
            graph.putIfAbsent(ticket.get(0), new LinkedList<>());
            graph.get(ticket.get(0)).add(ticket.get(1));
        }
        graph.values().forEach(x -> x.sort(String::compareTo));
        dfs(graph, "JFK", res);
        return res;
    }

    private static void dfs(Map<String, List<String>> graph, String src, List<String> res) {
        List<String> dest = graph.get(src);
        while (dest != null && dest.size() > 0) {
            //选择
            String det = dest.remove(0);
            dfs(graph, det, res);
        }
        //所有的遍历完，就是结束
        res.add(0, src);
    }

    //[386].字典序排数
    public static List<Integer> lexicalOrder(int n) {
        List<Integer> res = new ArrayList<>();
        for (int i = 1; i < 10; i++) {
            dfsForLexicalOrder(n, i, res);
        }
        return res;
    }

    private static void dfsForLexicalOrder(int n, int i, List<Integer> res) {
        if (n < i) {
            return;
        }
        res.add(i);
        for (int j = 0; j <= 9; j++) {
            dfsForLexicalOrder(n, i * 10 + j, res);
        }
    }

    //[401].二进制手表
    public static List<String> readBinaryWatch(int num) {
        List<String> res = new ArrayList<>();
        dfsForReadBinaryWatch(res, num, 0, 0, 1, 1, new LinkedList<>(), new LinkedList<>());
        return res;
    }

    private static void dfsForReadBinaryWatch(List<String> res, int num, int hour, int minute, int hstart, int mstart, LinkedList<Integer> hours, LinkedList<Integer> minutes) {
        if (hours.size() + minutes.size() == num) {
            if (hour < 12 && minute < 60) {
                res.add(String.format("%d:%02d", hour, minute));
            }
            return;
        }

        for (int i = hstart; i <= 8; i <<= 1) {
            hours.addLast(i);
            dfsForReadBinaryWatch(res, num, hour + i, minute, i << 1, mstart, hours, minutes);
            hours.removeLast();
        }

        for (int i = mstart; i <= 32; i <<= 1) {
            minutes.addLast(i);
            dfsForReadBinaryWatch(res, num, hour, minute + i, 16, i << 1, hours, minutes);
            minutes.removeLast();
        }
    }

    //[417].太平洋大西洋水流问题
    public static List<List<Integer>> pacificAtlantic(int[][] matrix) {

        List<List<Integer>> res = new ArrayList<>();
        if (matrix == null || matrix.length == 0) return res;
        int row = matrix.length;
        int col = matrix[0].length;
        boolean[][] visited = new boolean[row][col];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (dfsPacific(matrix, i, j, visited, Integer.MAX_VALUE)
                        && dfsAtlantic(matrix, i, j, visited, Integer.MAX_VALUE)) {
                    res.add(Arrays.asList(i, j));
                }
            }
        }
        return res;
    }

    private static boolean dfsPacific(int[][] matrix, int i, int j, boolean[][] visited, int pre) {
        int row = matrix.length;
        int col = matrix[0].length;
        if (i > row - 1 || j > col - 1 || visited[i][j] || matrix[i][j] > pre) {
            return false;
        }

        if (i <= 0 || j <= 0) {
            return true;
        }

        visited[i][j] = true;

        int cur = matrix[i][j];
        boolean result = dfsPacific(matrix, i - 1, j, visited, cur)
                || dfsPacific(matrix, i, j + 1, visited, cur)
                || dfsPacific(matrix, i + 1, j, visited, cur)
                || dfsPacific(matrix, i, j - 1, visited, cur);

        visited[i][j] = false;
        return result;
    }

    private static boolean dfsAtlantic(int[][] matrix, int i, int j, boolean[][] visited, int pre) {

        int row = matrix.length;
        int col = matrix[0].length;
        if (i < 0 || j < 0 || visited[i][j] || matrix[i][j] > pre) {
            return false;
        }

        if (i >= row - 1 || j >= col - 1) {
            return true;
        }

        visited[i][j] = true;

        int cur = matrix[i][j];
        boolean result = dfsAtlantic(matrix, i - 1, j, visited, cur)
                || dfsAtlantic(matrix, i, j + 1, visited, cur)
                || dfsAtlantic(matrix, i + 1, j, visited, cur)
                || dfsAtlantic(matrix, i, j - 1, visited, cur);

        visited[i][j] = false;
        return result;
    }

    //[433].最小基因变化
    public static int minMutation(String start, String end, String[] bank) {
        AtomicInteger res = new AtomicInteger(Integer.MAX_VALUE);
        dfsForMinMutation(new HashSet<>(), start, end, bank, new AtomicInteger(), res);
        return (res.get() == Integer.MAX_VALUE) ? -1 : res.get();
    }

    private static void dfsForMinMutation(Set<String> steps, String current, String end, String[] bank, AtomicInteger stepCount, AtomicInteger minCount) {
        //结束条件
        if (end.equals(current)) {
            minCount.set(Math.min(stepCount.get(), minCount.get()));
        }

        for (String next : bank) {
            //准备选择
            int diff = 0;
            for (int i = 0; i < next.length(); i++) {
                if (next.charAt(i) != current.charAt(i)) {
                    if (++diff > 1) {
                        break;
                    }
                }
            }
            //做选择，没有遍历过，并且字符相差1
            if (diff == 1 && !steps.contains(next)) {
                steps.add(next);
                stepCount.incrementAndGet();
                dfsForMinMutation(steps, next, end, bank, stepCount, minCount);
                stepCount.decrementAndGet();
                steps.remove(next);
            }
        }
    }

    //[464].我能赢吗
    public static boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        if (maxChoosableInteger >= desiredTotal) return true;
        //累加和比所有和还要小，都赢不了
        if ((1 + maxChoosableInteger) * maxChoosableInteger / 2 < desiredTotal) return false;
        int[] select = new int[maxChoosableInteger + 1];
        return dfsForCanIWin(select, desiredTotal, new HashMap<>());
    }

    private static boolean dfsForCanIWin(int[] select, int desiredTotal, Map<String, Boolean> visited) {
        String key = Arrays.toString(select);
        if (visited.containsKey(key)) return visited.get(key);
        //选择
        for (int i = 1; i < select.length; i++) {
            //还没有选择
            if (select[i] == 0) {
                select[i] = 1;

                //当前的数字比期望值大(赢了) + 下个玩家从剩下的期望值不能获胜
                if (desiredTotal <= i || !dfsForCanIWin(select, desiredTotal - i, visited)) {
                    visited.put(key, true);
                    select[i] = 0;
                    return true;
                }
                select[i] = 0;
            }
        }
        visited.put(key, false);
        return false;
    }

    //[473].火柴拼正方形
    public static boolean makesquare(int[] nums) {
        if (null == nums || nums.length < 4) {
            return false;
        }
        int sum = 0;
        for (int num : nums) {
            sum = sum + num;
        }
        if (sum % 4 != 0) {
            return false;
        }

        return dfsForMakesquare(nums, 0, 0, 0, 0, 0, sum / 4);
    }

    private static boolean dfsForMakesquare(int[] nums, int index, int a, int b, int c, int d, int side) {
        if (nums.length == index) {
            return a == b && b == c && c == d && d == a;
        }
        if (a > side || b > side || c > side || d > side) {
            return false;
        }

        int num = nums[index];
        return dfsForMakesquare(nums, index + 1, a + num, b, c, d, side)
                || dfsForMakesquare(nums, index + 1, a, b + num, c, d, side)
                || dfsForMakesquare(nums, index + 1, a, b, c + num, d, side)
                || dfsForMakesquare(nums, index + 1, a, b, c, d + num, side);
    }

    //[491].递增子序列
    public static List<List<Integer>> findSubsequences(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        LinkedList select = new LinkedList();
        dfs(nums, -1, res, select);
        return res;
    }

    private static void dfs(int[] nums, int start, List<List<Integer>> res, LinkedList<Integer> select) {
        if (select.size() > 1) {
            res.add(new ArrayList<>(select));
        }

        Set<Integer> set = new HashSet<>();
        for (int i = start + 1; i < nums.length; i++) {
            if (set.contains(nums[i])) {
                continue;
            }

            set.add(nums[i]);
            if (start == -1 || nums[i] >= nums[start]) {
                select.add(nums[i]);
                dfs(nums, i, res, select);
                select.removeLast();
            }
        }
    }


    //[494].目标和
    public static int findTargetSumWays(int[] nums, int S) {
        AtomicInteger res = new AtomicInteger();
        dfs(nums, 0, S, res);
        return res.get();
    }

    private static void dfs(int[] nums, int index, int target, AtomicInteger res) {
        if (index == nums.length) {
            if (target == 0) {
                res.incrementAndGet();
            }
            return;
        }

        target += nums[index];
        dfs(nums, index + 1, target, res);
        target -= nums[index];

        target -= nums[index];
        dfs(nums, index + 1, target, res);
        target += nums[index];
    }
    //[529].扫雷机器人
    public static char[][] updateBoard(char[][] board, int[] click) {
        int x = click[0], y = click[1];
        int[][] directs = new int[][]{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
        if (board[x][y] == 'M') {
            board[x][y] = 'X';
        } else {
            dfsForUpdateBoard(board, x, y, directs);
        }
        return board;
    }

    private static void dfsForUpdateBoard(char[][] board, int x, int y, int[][] directs) {
        int bomb = 0;
        for (int[] direct : directs) {
            int newX = x + direct[0];
            int newY = y + direct[1];
            if (newX < 0 || newY < 0 || newX >= board.length || newY >= board[0].length) {
                continue;
            }
            if (board[newX][newY] == 'M') {
                bomb++;
            }
        }
        board[x][y] = bomb > 0 ? (char) (bomb + '0') : 'B';
        if (board[x][y] == 'B') {
            for (int i = 0; i < directs.length; i++) {
                int newX = x + directs[i][0];
                int newY = y + directs[i][1];
                if (newX < 0 || newY < 0 || newX >= board.length || newY >= board[0].length || board[newX][newY] != 'E') {
                    continue;
                }
                dfsForUpdateBoard(board, newX, newY, directs);
            }
        }
    }

    //[542].01 矩阵
    public static int[][] updateMatrix(int[][] matrix) {
        int row = matrix.length;
        int col = matrix[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                //设置周边都不是0的1的点，然后更改。
                if (matrix[i][j] == 1 &&
                        !((i > 0 && matrix[i - 1][j] == 0)
                                || (j > 0 && matrix[i][j - 1] == 0)
                                || (i < row - 1 && matrix[i + 1][j] == 0)
                                || (j < col - 1 && matrix[i][j + 1] == 0))) {
                    matrix[i][j] = Integer.MAX_VALUE;
                }
            }
        }

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                //只需递归遍历为1的数字
                if (matrix[i][j] == 1) {
                    dfsForUpdateMatrix(matrix, i, j);
                }
            }
        }
        return matrix;
    }

    private static void dfsForUpdateMatrix(int[][] matrix, int x, int y) {
        int[][] direct = new int[][]{{1, 0}, {-1, 0}, {0, -1}, {0, 1}};
        for (int[] dir : direct) {
            int newX = x + dir[0];
            int newY = y + dir[1];
            if (newX >= 0 && newX < matrix.length
                    && newY >= 0 && newY < matrix[0].length
                    && matrix[newX][newY] > matrix[x][y] + 1) {
                matrix[newX][newY] = matrix[x][y] + 1;
                dfsForUpdateMatrix(matrix, newX, newY);
            }
        }
    }

    public static void main(String[] args) {
//        [17].电话号码的字母组合
//        letterCombinations("23");
//        letterCombinations("");
//
//        [22].括号生成
//        System.out.println(generateParenthesis(0));
//
//        [39].组合总和
//        combinationSum(new int[]{2, 3, 6, 7}, 7);

//        [40].组合总和II
//        combinationSum2(new int[]{10,1,2,7,6,1,5}, 8);
//        combinationSum2(new int[]{2,5,2,1,2}, 5);
//
//        [46].全排列
//        System.out.println(permute(new int[]{1, 2, 3}));
//
//        [47].全排列II
//        permuteUnique(new int[]{1, 1, 2});
//        permuteUnique(new int[]{3, 3, 0, 3});
//
//        [51].N皇后
//        System.out.println(solveNQueens(8));
//
//        [77].组合
//        System.out.println(combine(4, 2));
//        System.out.println(combine(2, 2));
//
//        [78].子集
//        System.out.println(subsets(new int[]{1, 2, 3}));
//
//        [79].单词搜索
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
//        [90].子集II
//        System.out.println(subsetsWithDup(new int[]{1, 2, 2, 3}));
        System.out.println(subsetsWithDup(new int[]{1, 2, 2}));
//        System.out.println(subsetsWithDup(new int[]{}));
//
//        [131].分割回文串
//        System.out.println(partition("aba"));
//        System.out.println(partition("aab"));
//        System.out.println(partition("aabbaacbc"));
//
//        [216].组合总数III
//        System.out.println(combinationSum3(3, 9));
//        System.out.println(combinationSum3(3, 7));
//        System.out.println(combinationSum3(3, 1));
//        System.out.println(combinationSum3(1, -1));
//
//        [306].累加数
//        System.out.println(isAdditiveNumber("11"));
//        System.out.println(isAdditiveNumber(""));
//        System.out.println(isAdditiveNumber("110"));
//        System.out.println(isAdditiveNumber("112"));
//        System.out.println(isAdditiveNumber("199100199"));
//
//        [332].重新安排行程
//        System.out.println(findItinerary(Arrays.asList(Arrays.asList("JFK", "SFO"),
//        Arrays.asList("JFK", "ATL"),
//                Arrays.asList("SFO", "ATL"),
//                Arrays.asList("ATL", "JFK"),
//                Arrays.asList("ATL", "SFO"))));

//        System.out.println(lexicalOrder(9));
//        System.out.println(lexicalOrder(13));
//        System.out.println(lexicalOrder(2002));
//
//        [401].二进制手表
//        System.out.println(readBinaryWatch(2));
//        System.out.println(readBinaryWatch(7));
//
//        [417].太平洋大西洋水流问题
//        System.out.println(pacificAtlantic(new int[][]{{1,2,2,3,5},{3,2,3,4,4},{2,4,5,3,1},{6,7,1,4,5},{5,1,1,2,4}}));
//
//        [433].最小基因变化
//        System.out.println(minMutation("AAAAACCC", "AACCCCCC", new String[]{"AAAACCCC", "AAACCCCC", "AACCCCCC"}));
//        System.out.println(minMutation("AACCGGTT", "AAACGGTA", new String[]{"AACCGGTA", "AACCGCTA", "AAACGGTA"}));
//
//        [473].火柴拼正方形
//        System.out.println(makesquare(new int[]{1, 1, 2, 2, 2}));
//        System.out.println(makesquare(new int[]{3, 3, 3, 3, 4}));
//
//        [491].递增子序列
//        System.out.println(findSubsequences(new int[]{4, 6, 7, 7}));
//
//        [494].目标和
//        System.out.println(findTargetSumWays(new int[]{1, 1, 1, 1, 1}, 3));
//
//        [529].扫雷机器人
//        char[][] board = new char[][]{{'E', 'E', 'E', 'E', 'E'}, {'E', 'E', 'M', 'E', 'E'}, {'E', 'E', 'E', 'E', 'E'}, {'E', 'E', 'E', 'E', 'E'}};
//        updateBoard(board, new int[]{3, 0});
//        updateBoard(board, new int[]{1, 2});
//
//        [542].01 矩阵
//        int[][] res = updateMatrix(new int[][]{{0,0,0},{0,1,0},{1,1,1}});
//        System.out.println();

    }
}
