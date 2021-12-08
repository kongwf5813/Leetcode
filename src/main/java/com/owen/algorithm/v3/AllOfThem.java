package com.owen.algorithm.v3;


import com.owen.algorithm.Tree;

import javax.swing.*;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class AllOfThem {
    public static class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        public ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    public static class TreeNode {
        public int val;
        public TreeNode left;
        public TreeNode right;

        public TreeNode(int x) {
            val = x;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public static class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;
        public Node random;
        public List<Node> neighbors;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }

    //[2].两数相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int capacity = 0;
        ListNode dummyHead = new ListNode(-1);
        ListNode cur = dummyHead;
        while (l1 != null || l2 != null || capacity != 0) {
            int v1 = 0;
            if (l1 != null) {
                v1 = l1.val;
                l1 = l1.next;
            }
            int v2 = 0;
            if (l2 != null) {
                v2 = l2.val;
                l2 = l2.next;
            }
            int val = v1 + v2 + capacity;
            cur.next = new ListNode(val % 10);
            cur = cur.next;
            capacity = val / 10;
        }
        return dummyHead.next;
    }

    //[3].无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> window = new HashMap<>();
        int left = 0, right = 0, res = 0;
        while (right < s.length()) {
            char r = s.charAt(right);
            right++;
            window.put(r, window.getOrDefault(r, 0) + 1);
            while (window.get(r) > 1) {
                char l = s.charAt(left);
                window.put(l, window.get(l) - 1);
                left++;
            }
            //窗口扩大的时候求最长子串
            res = Math.max(res, right - left);
        }
        return res;
    }

    //[11].盛最多水的容器
    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1, area = 0;
        while (left < right) {
            area = Math.max(area, Math.min(height[right], height[left]) * (right - left));
            //那边最短，往里面缩
            if (height[left] > height[right]) {
                right--;
            } else {
                left++;
            }
        }
        return area;
    }

    //[22].括号生成
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        if (n <= 0) return res;
        backtraceForGenerateParenthesis(n, n, res, new StringBuilder());
        return res;
    }

    private void backtraceForGenerateParenthesis(int left, int right, List<String> res, StringBuilder sb) {
        if (right < left || left < 0 || right < 0) {
            return;
        }
        if (left == 0 && right == 0) {
            res.add(sb.toString());
            return;
        }

        sb.append('(');
        backtraceForGenerateParenthesis(left - 1, right, res, sb);
        sb.deleteCharAt(sb.length() - 1);

        sb.append(')');
        backtraceForGenerateParenthesis(left, right - 1, res, sb);
        sb.deleteCharAt(sb.length() - 1);
    }

    //[46].全排列
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        LinkedList<Integer> path = new LinkedList<>();
        dfsForPermute(res, path, nums);
        return res;
    }

    private void dfsForPermute(List<List<Integer>> res, LinkedList<Integer> path, int[] nums) {
        if (path.size() == nums.length) {
            res.add(new LinkedList<>(path));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            if (path.contains(nums[i])) {
                continue;
            }
            path.addLast(nums[i]);
            dfsForPermute(res, path, nums);
            path.removeLast();
        }
    }

    //[47].全排列 II
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length == 0) {
            return res;
        }
        Arrays.sort(nums);
        LinkedList<Integer> path = new LinkedList<>();
        boolean[] visited = new boolean[nums.length];
        dfsForPermuteUnique(res, path, nums, visited);
        return res;
    }

    private void dfsForPermuteUnique(List<List<Integer>> res, LinkedList<Integer> path, int[] nums, boolean[] visited) {
        if (path.size() == nums.length) {
            res.add(new LinkedList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i]) {
                continue;
            }
            //决策树画完之后，发现01这种状态需要剪枝，意思是重复的数。
            //一定从左边往右边选: 如果左边的还没有选，则右边的也不选，直接跳过。
            if (i > 0 && nums[i] == nums[i - 1] && !visited[i - 1]) {
                continue;
            }
            visited[i] = true;
            path.add(nums[i]);
            dfsForPermuteUnique(res, path, nums, visited);
            path.removeLast();
            visited[i] = false;
        }
    }

    //[62].不同路径
    public int uniquePaths(int m, int n) {
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        //          (i-1,j)
        //             ↓
        //(i,j-1) →  (i,j)
        //遍历顺序是从左往右，垂直投影，砍掉i维度之后，dp[i-1][j]的值就是之前的dp[j]
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[j] = dp[j] + dp[j - 1];
            }
        }
        return dp[n - 1];
    }

    //[207].课程表
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        //入度，找到入度为0的节点，然后依次遍历，减度数，如果为入度为0加入
        int[] indegree = new int[numCourses];
        for (int[] pre : prerequisites) {
            indegree[pre[0]]++;
        }

        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < indegree.length; i++) {
            if (indegree[i] == 0) {
                queue.offer(i);
            }
        }
        int count = 0;
        while (!queue.isEmpty()) {
            int course = queue.poll();
            count++;
            for (int[] pre : prerequisites) {
                if (pre[1] != course) continue;
                indegree[pre[0]]--;

                //只有入度为0的点才加进去
                if (indegree[pre[0]] == 0) {
                    queue.offer(pre[0]);
                }
            }
        }
        return count == numCourses;
    }

    //[210].课程表 II
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] indeg = new int[numCourses];
        for (int[] pre : prerequisites) {
            indeg[pre[0]]++;
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indeg[i] == 0) {
                queue.offer(i);
            }
        }

        int[] res = new int[numCourses];
        int index = 0;
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            res[index++] = cur;
            for (int[] pre : prerequisites) {
                if (pre[1] != cur) continue;
                if (--indeg[pre[0]] == 0) {
                    queue.offer(pre[0]);
                }
            }

        }
        return index == numCourses ? res : new int[0];
    }

    //[310].最小高度树
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if (n == 1) {
            return Arrays.asList(0);
        }
        int[] deg = new int[n];
        Map<Integer, List<Integer>> link = new HashMap<>();
        for (int[] edge : edges) {
            deg[edge[0]]++;
            deg[edge[1]]++;
            link.putIfAbsent(edge[0], new ArrayList<>());
            link.get(edge[0]).add(edge[1]);

            link.putIfAbsent(edge[1], new ArrayList<>());
            link.get(edge[1]).add(edge[0]);
        }
        Queue<Integer> queue = new LinkedList<>();
        //无向图的出度判断为1，有向图出度为0
        //叶子节点的出度为1
        for (int i = 0; i < deg.length; i++) {
            if (deg[i] == 1) {
                queue.offer(i);
            }
        }
        List<Integer> res = new ArrayList<>();
        while (!queue.isEmpty()) {
            int size = queue.size();
            res = new ArrayList<>();
            //最外面的一层，叶子节点全部移除.
            for (int i = 0; i < size; i++) {
                int cur = queue.poll();
                List<Integer> neighbours = link.get(cur);
                for (int neighbour : neighbours) {
                    if (--deg[neighbour] == 1) {
                        queue.offer(neighbour);
                    }
                }
                res.add(cur);
            }
        }
        return res;
    }

    //[326].3的幂
    public boolean isPowerOfThree(int n) {
        if (n <= 0) return false;
        // 45 = 3 * 3 * 5
        // 9 = 3 * 3 * 1
/*        while (n % 3 == 0) {
            n = n / 3;
        }
        return n == 1;*/

        //因为3是质数，所以3^19肯定是3^n的最大公约数。
        int max = (int) Math.pow(3, 19);
        return max % n == 0;
    }

    //[342].4的幂
    public boolean isPowerOfFour(int n) {
        // 16 = 4*4*1
        if (n <= 0) return false;
        //2的幂， 而且n & 1010 1010 1010 1010 为0，偶数位为1
        return (n & (n - 1)) == 0 && (n & 0xaaaaaaaa) == 0;
    }

    //[743].网络延迟时间
    public int networkDelayTime(int[][] times, int n, int k) {
        //最短路径
        int[] distTo = dijkstra(times, k, n);
        int res = 0;
        for (int i = 1; i <= n; i++) {
            if (distTo[i] == Integer.MAX_VALUE) {
                return -1;
            }
            res = Math.max(res, distTo[i]);
        }
        return res;
    }

    private int[] dijkstra(int[][] times, int start, int n) {
        List<int[]>[] graph = new LinkedList[n + 1];
        for (int i = 1; i < n + 1; i++) {
            graph[i] = new LinkedList<>();
        }
        for (int[] time : times) {
            graph[time[0]].add(new int[]{time[1], time[2]});
        }

        int[] distTo = new int[n + 1];
        Arrays.fill(distTo, Integer.MAX_VALUE);
        //id + 目前的最短距离
        Queue<int[]> queue = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
        queue.offer(new int[]{start, 0});
        distTo[start] = 0;
        while (!queue.isEmpty()) {
            int[] curNode = queue.poll();
            int id = curNode[0];
            int dist = curNode[1];
            if (dist > distTo[id]) {
                continue;
            }
            for (int[] next : graph[start]) {
                int nextId = next[1];
                int weight = next[2];
                int distToNext = distTo[id] + weight;
                if (distToNext < distTo[nextId]) {
                    distTo[nextId] = distToNext;
                    queue.offer(new int[]{nextId, distToNext});
                }
            }
        }
        return distTo;
    }

    //[261].以图判树
    public boolean validTree(int n, int[][] edges) {
        UnionFind uf = new UnionFind(n);
        for (int[] edge : edges) {
            if (uf.connect(edge[0], edge[1])) {
                return false;
            }
            uf.union(edge[0], edge[1]);
        }
        return uf.count == 1;
    }

    //[1135].最低成本联通所有城市
    public int minimumCost(int N, int[][] connections) {
        UnionFind uf = new UnionFind(N + 1);
        //最低成本升序
        Arrays.sort(connections, (a, b) -> a[2] - b[2]);
        int res = Integer.MAX_VALUE;
        for (int[] connect : connections) {
            if (uf.connect(connect[0], connect[1])) {
                continue;
            }
            uf.union(connect[0], connect[1]);
            res += connect[2];
        }
        return uf.count == 2 ? res : -1;
    }

    //[1584].连接所有点的最小费用
    public int minCostConnectPoints(int[][] points) {
        int size = points.length;
        //点要转化为边集合
        List<int[]> edges = new ArrayList<>();
        for (int i = 0; i < size - 1; i++) {
            for (int j = i + 1; j < size; j++) {
                edges.add(new int[]{i, j, Math.abs(points[i][0] - points[j][0]) + Math.abs(points[i][1] - points[j][1])});
            }
        }

        Collections.sort(edges, Comparator.comparingInt(a -> ((int[]) a)[2]));

        int res = 0;
        UnionFind uf = new UnionFind(size);
        for (int[] edge : edges) {
            if (uf.connect(edge[0], edge[1])) {
                continue;
            }
            res += edge[2];

            uf.union(edge[0], edge[1]);
        }
        return res;
    }

    public static class UnionFind {
        private int[] parent;
        private int[] size;
        private int count;

        public UnionFind(int count) {
            parent = new int[count];
            size = new int[count];
            this.count = count;
            for (int i = 0; i < count; i++) {
                parent[i] = i;
                size[i] = 1;
            }
        }

        public int find(int x) {
            while (x != parent[x]) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }

        public boolean connect(int p, int q) {
            return find(p) == find(q);
        }

        public void union(int p, int q) {
            int rootP = find(p);
            int rootQ = find(q);
            if (rootP == rootQ) {
                return;
            }
            if (size[rootP] > size[rootQ]) {
                parent[rootQ] = rootP;
                size[rootP] += size[rootQ];
            } else {
                parent[rootP] = rootQ;
                size[rootQ] += size[rootP];
            }
            count--;
        }
    }

    //[130].被围绕的区域
    public void solve(char[][] board) {
        int m = board.length, n = board[0].length;
        //从边界出发找到O的替换掉
        for (int i = 0; i < m; i++) {
            dfsForSolve(board, i, 0);
            dfsForSolve(board, i, n - 1);
        }
        for (int j = 0; j < m; j++) {
            dfsForSolve(board, 0, j);
            dfsForSolve(board, m - 1, j);
        }
        //把边界的O修改成X
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                } else if (board[i][j] == 'Y') {
                    board[i][j] = 'O';
                }
            }
        }

    }

    private void dfsForSolve(char[][] board, int x, int y) {
        int m = board.length, n = board[0].length;
        if (x < 0 || y < 0 || x > m - 1 || y > n - 1) {
            return;
        }
        //不是水
        if (board[x][y] != 'O') {
            return;
        }
        //淹掉它
        board[x][y] = 'Y';
        dfsForSolve(board, x - 1, y);
        dfsForSolve(board, x + 1, y);
        dfsForSolve(board, x, y - 1);
        dfsForSolve(board, x, y + 1);
    }

    //[200].岛屿数量
    public int numIslands(char[][] grid) {
        int m = grid.length, n = grid[0].length;
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    res++;
                    dfsForIslands(grid, i, j);
                }
            }
        }
        return res;
    }

    private void dfsForIslands(char[][] grid, int x, int y) {
        int m = grid.length, n = grid[0].length;
        if (x < 0 || x > m - 1 || y < 0 || y > n - 1) {
            return;
        }

        if (grid[x][y] == '0') {
            return;
        }
        //是陆地就淹掉它
        grid[x][y] = '0';
        dfsForIslands(grid, x - 1, y);
        dfsForIslands(grid, x + 1, y);
        dfsForIslands(grid, x, y - 1);
        dfsForIslands(grid, x, y + 1);
    }

    //[463].岛屿的周长
    public int islandPerimeter(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        //1是岛屿，0是水
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    //题目说了只有一个
                    return dfsForIslandPerimeter(grid, i, j);
                }
            }
        }
        return 0;
    }

    private int dfsForIslandPerimeter(int[][] grid, int x, int y) {
        int m = grid.length, n = grid[0].length;
        //遇到边界就+1
        if (x < 0 || y < 0 || x >= m || y >= n) {
            return 1;
        }
        //遇到水域+1
        if (grid[x][y] == 0) {
            return 1;
        }
        //已经遍历过了，计数为0
        if (grid[x][y] == 2) {
            return 0;
        }
        grid[x][y] = 2;
        return dfsForIslandPerimeter(grid, x - 1, y)
                + dfsForIslandPerimeter(grid, x + 1, y)
                + dfsForIslandPerimeter(grid, x, y - 1)
                + dfsForIslandPerimeter(grid, x, y + 1);
    }

    //[695].岛屿的最大面积
    public int maxAreaOfIsland(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    int area = dfsForMaxAreaOfIsland(grid, i, j);
                    res = Math.max(res, area);
                }
            }
        }
        return res;
    }

    private int dfsForMaxAreaOfIsland(int[][] grid, int x, int y) {
        int m = grid.length, n = grid[0].length;
        if (x < 0 || y < 0 || x >= m || y >= n) {
            return 0;
        }
        if (grid[x][y] == 0) {
            return 0;
        }
        grid[x][y] = 0;
        return dfsForMaxAreaOfIsland(grid, x - 1, y)
                + dfsForMaxAreaOfIsland(grid, x + 1, y)
                + dfsForMaxAreaOfIsland(grid, x, y - 1)
                + dfsForMaxAreaOfIsland(grid, x, y + 1) + 1;
    }

    //[1254].统计封闭岛屿的数目
    public int closedIsland(int[][] grid) {
        //0是陆地，1是水
        int m = grid.length, n = grid[0].length;
        //先淹掉周边的陆地，剩下的都是被水包围的
        for (int i = 0; i < m; i++) {
            dfsForClosedIsland(grid, i, 0);
            dfsForClosedIsland(grid, i, n - 1);
        }
        for (int j = 0; j < n; j++) {
            dfsForClosedIsland(grid, 0, j);
            dfsForClosedIsland(grid, m - 1, j);
        }

        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    res++;
                    dfsForClosedIsland(grid, i, j);
                }
            }
        }
        return res;
    }

    private void dfsForClosedIsland(int[][] grid, int x, int y) {
        int m = grid.length, n = grid[0].length;
        if (x < 0 || y < 0 || x >= m || y >= n) {
            return;
        }
        if (grid[x][y] == 1) {
            return;
        }
        //淹掉它
        grid[x][y] = 1;

        dfsForClosedIsland(grid, x - 1, y);
        dfsForClosedIsland(grid, x + 1, y);
        dfsForClosedIsland(grid, x, y - 1);
        dfsForClosedIsland(grid, x, y + 1);
    }

    //[1905].统计子岛屿
    public int countSubIslands(int[][] grid1, int[][] grid2) {
        int m = grid1.length, n = grid1[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                //前面是水，后面是陆，淹掉它
                if (grid1[i][j] == 0 && grid2[i][j] == 1) {
                    dfsForCountSubIslands(grid2, i, j);
                }
            }
        }
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                //剩下的都应该是子集
                if (grid2[i][j] == 1) {
                    res++;
                    dfsForCountSubIslands(grid2, i, j);
                }
            }
        }
        return res;
    }

    private void dfsForCountSubIslands(int[][] grid, int x, int y) {
        int m = grid.length, n = grid[0].length;
        if (x < 0 || x > m - 1 || y < 0 || y > n - 1) {
            return;
        }

        if (grid[x][y] == 0) {
            return;
        }
        grid[x][y] = 0;

        dfsForCountSubIslands(grid, x - 1, y);
        dfsForCountSubIslands(grid, x + 1, y);
        dfsForCountSubIslands(grid, x, y - 1);
        dfsForCountSubIslands(grid, x, y + 1);
    }

    //[1034].边界着色
    public int[][] colorBorder(int[][] grid, int row, int col, int color) {
        //原始颜色是跟color可能不相同的
        //遇到的边界其实是只有看到下一个是颜色不同的时候，才能确定的
        //岛屿可以任意修改，要么0，要么1，可以通过修改达到访问的目的
        //只更改边界颜色，意味着访问了之后不能更改颜色，不然回溯之后条件变化会导致判断条件失败，所以保存起来，之后再修改。
        int m = grid.length, n = grid[0].length;
        LinkedList<int[]> border = new LinkedList<>();
        dfsForColorBorder(grid, row, col, grid[row][col], border, new boolean[m][n]);

        for (int[] need : border) {
            grid[need[0]][need[1]] = color;
        }
        return grid;
    }

    private void dfsForColorBorder(int[][] grid, int x, int y, int originalColor, LinkedList<int[]> border, boolean[][] visit) {
        int m = grid.length, n = grid[0].length;
        int[][] directions = new int[][]{{0, -1}, {0, 1}, {-1, 0}, {1, 0}};

        visit[x][y] = true;
        boolean isBorder = false;
        for (int[] direct : directions) {
            int nx = x + direct[0];
            int ny = y + direct[1];

            if (!(nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == originalColor)) {
                isBorder = true;
            } else if (!visit[nx][ny]) {
                dfsForColorBorder(grid, nx, ny, originalColor, border, visit);
            }
        }
        if (isBorder) {
            border.addLast(new int[]{x, y});
        }
    }

    private int[][] colorBorderV2(int[][] grid, int row, int col, int color) {
        int m = grid.length, n = grid[0].length;
        dfsForColorBorder(grid, row, col, color, grid[row][col], new boolean[m][n]);
        return grid;
    }

    private void dfsForColorBorder(int[][] grid, int x, int y, int color, int originalColor, boolean[][] visit) {
        int m = grid.length, n = grid[0].length;
        int[][] directions = new int[][]{{0, -1}, {0, 1}, {-1, 0}, {1, 0}};

        visit[x][y] = true;
        for (int[] direct : directions) {
            int nx = x + direct[0];
            int ny = y + direct[1];
            //访问过的，可能处理过边界，所以不访问了
            if (nx >= 0 && ny >= 0 && nx < m && ny < n && !visit[nx][ny] && grid[nx][ny] == originalColor) {
                //找到跟原始相同的颜色，不是边界，则继续递归
                dfsForColorBorder(grid, nx, ny, color, originalColor, visit);
            } else if (nx >= 0 && ny >= 0 && nx < m && ny < n && !visit[nx][ny] && grid[nx][ny] != originalColor || nx < 0 || ny < 0 || nx >= m || ny >= n) {
                //找到跟原始不相同的颜色，或者超出边界，直接修改颜色
                grid[x][y] = color;
            }
        }
    }


    public List<TreeNode> generateTrees(int n) {
        if (n == 0) return null;
        return dfsForGenerateTrees(1, n);
    }

    private List<TreeNode> dfsForGenerateTrees(int start, int end) {
        List<TreeNode> res = new ArrayList<>();
        if (start > end) {
            res.add(null);
            return res;
        }
        for (int i = start; i <= end; i++) {
            List<TreeNode> leftTree = dfsForGenerateTrees(start, i - 1);
            List<TreeNode> rightTree = dfsForGenerateTrees(i + 1, end);
            //始终都只有一个节点
            for (TreeNode left : leftTree) {
                for (TreeNode right : rightTree) {
                    TreeNode root = new TreeNode(i);
                    root.left = left;
                    root.right = right;
                    res.add(root);
                }
            }
        }
        return res;
    }

    //[96].不同的二叉搜索树
    public int numTrees(int n) {
        //方法1 递归解决
        //return dfsForNumTrees(new int[n + 1][n + 1], 1, n);

        //方法2
        // 1  1
        // 2  2
        // 3  2 + 1 + 2
        // 4  5 + 1 * 2 + 2*1 + 5
        //i个节点的数量，而不是前i个节点的数量
        //dp[i] = dp[0] * dp [i - 1] + dp[1] * dp[i-2] + ... + dp[i-1]* dp[0]
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                dp[i] += dp[j] * dp[i - j - 1];
            }
        }
        return dp[n];
    }

    private int dfsForNumTrees(int[][] memo, int start, int end) {
        if (start > end) {
            return 1;
        }
        if (memo[start][end] != 0) {
            return memo[start][end];
        }
        int res = 0;
        for (int i = start; i <= end; i++) {
            int left = dfsForNumTrees(memo, start, i - 1);
            int right = dfsForNumTrees(memo, i + 1, end);
            res += left * right;
        }
        memo[start][end] = res;
        return res;
    }

    //[98].验证二叉搜索树
    public boolean isValidBST(TreeNode root) {
        return dfsIsValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    private boolean dfsIsValidBST(TreeNode root, Long min, Long max) {
        if (root == null) return true;
        if (min < root.val && root.val < max) {
            return dfsIsValidBST(root.left, min, (long) root.val) &&
                    dfsIsValidBST(root.right, (long) root.val, max);
        }
        return false;
    }

    //[100].相同的树
    public boolean isSameTree(TreeNode p, TreeNode q) {
        return dfsIsSameTree(p, q);
    }

    private boolean dfsIsSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null && q != null) return false;
        if (p != null && q == null) return false;
        if (p.val != q.val) return false;
        return dfsIsSameTree(p.left, q.left) && dfsIsSameTree(p.right, q.right);
    }

    //[101].对称二叉树
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return dfsIsSymmetric(root.left, root.right);
    }

    private boolean dfsIsSymmetric(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        //是否为镜像，意味着，主节点相等， p的左节点 = q的右节点， p的右节点 = q的左节点
        if (p == null && q != null) return false;
        if (p != null && q == null) return false;
        if (p.val != q.val) return false;
        return dfsIsSymmetric(p.left, q.right) && dfsIsSymmetric(p.right, q.left);
    }

    //[102].二叉树的层序遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) return null;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        List<List<Integer>> res = new ArrayList<>();
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> level = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();
                level.add(cur.val);

                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }
            }
            res.add(level);
        }
        return res;
    }

    //[103].二叉树的锯齿形层序遍历
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        if (root == null) return null;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean flag = true;
        List<List<Integer>> res = new ArrayList<>();
        while (!queue.isEmpty()) {
            LinkedList<Integer> list = new LinkedList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();
                if (flag) {
                    list.addLast(cur.val);
                } else {
                    list.addFirst(cur.val);
                }

                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }
            }
            res.add(list);
            flag = !flag;
        }
        return res;
    }

    //[104].二叉树的最大深度
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    //[105].从前序与中序遍历序列构造二叉树
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int pLen = preorder.length;
        int iLen = inorder.length;
        if (pLen != iLen) {
            return null;
        }
        Map<Integer, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < iLen; i++) {
            indexMap.put(inorder[i], i);
        }
        return dfsForBuildTree(preorder, inorder, 0, pLen - 1, 0, iLen - 1, indexMap);
    }

    private TreeNode dfsForBuildTree(int[] preorder, int[] inorder, int ps, int pe, int is, int ie, Map<Integer, Integer> indexMap) {
        if (ps > pe || is > ie) return null;
        int rootVal = preorder[ps];
        TreeNode root = new TreeNode(rootVal);
        int index = indexMap.get(rootVal);
        TreeNode left = dfsForBuildTree(preorder, inorder, ps + 1, index - is + ps, is, index - 1, indexMap);
        TreeNode right = dfsForBuildTree(preorder, inorder, index - is + ps + 1, pe, index + 1, ie, indexMap);
        root.left = left;
        root.right = right;
        return root;
    }

    //[106].从中序与后序遍历序列构造二叉树
    public TreeNode buildTree2(int[] inorder, int[] postorder) {
        int iLen = inorder.length;
        int pLen = postorder.length;
        if (iLen != pLen) return null;
        Map<Integer, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < iLen; i++) {
            indexMap.put(inorder[i], i);
        }
        return dfsForBuildTree2(inorder, postorder, 0, iLen - 1, 0, pLen - 1, indexMap);
    }

    private TreeNode dfsForBuildTree2(int[] inorder, int[] postorder, int is, int ie, int ps, int pe, Map<Integer, Integer> indexMap) {
        if (is > ie || ps > pe) {
            return null;
        }
        int rootVal = postorder[pe];
        TreeNode root = new TreeNode(rootVal);
        int index = indexMap.get(rootVal);

        TreeNode left = dfsForBuildTree2(inorder, postorder, is, index - 1, ps, ps + index - is - 1, indexMap);
        TreeNode right = dfsForBuildTree2(inorder, postorder, index + 1, ie, ps + index - is, pe - 1, indexMap);
        root.left = left;
        root.right = right;
        return root;
    }

    //[107].二叉树的层序遍历 II
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> res = new LinkedList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if (root != null) {
            queue.offer(root);
        }
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> level = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode cur = queue.poll();
                level.add(cur.val);

                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }
            }
            res.add(0, level);
        }
        return res;
    }

    //[108].将有序数组转换为二叉搜索树
    public TreeNode sortedArrayToBST(int[] nums) {
        return dfsForSortedArrayToBST(nums, 0, nums.length - 1);
    }

    private TreeNode dfsForSortedArrayToBST(int[] nums, int s, int e) {
        if (s > e) return null;
        int mid = s + (e - s) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        TreeNode left = dfsForSortedArrayToBST(nums, s, mid - 1);
        TreeNode right = dfsForSortedArrayToBST(nums, mid + 1, e);
        root.left = left;
        root.right = right;
        return root;
    }

    //[109].有序链表转换二叉搜索树
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) return null;
        ListNode slow = head, fast = head, pre = null;
        while (fast != null && fast.next != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        TreeNode root = new TreeNode(slow.val);
        //只有一个节点了
        if (pre == null) return root;

        //断开前面一个节点
        pre.next = null;
        root.left = sortedListToBST(head);
        root.right = sortedListToBST(slow.next);
        return root;
    }

    //[110].平衡二叉树
    public boolean isBalanced(TreeNode root) {
        return dfsForIsBalanced(root) >= 0;
    }

    //不平衡就是-1， 否则返回高度
    private int dfsForIsBalanced(TreeNode root) {
        if (root == null) return 0;
        int left = dfsForIsBalanced(root.left);
        int right = dfsForIsBalanced(root.right);

        if (left == -1 || right == -1 || Math.abs(left - right) > 1) {
            return -1;
        }
        return Math.max(left, right) + 1;
    }

    //[111].二叉树的最小深度
    public int minDepth(TreeNode root) {
        if (root == null) return 0;
        int left = minDepth(root.left);
        int right = minDepth(root.right);
        //有一棵子树没有，此时高度应该是2， 而不是下面的1，计算就会不对。
        if (left == 0 || right == 0) {
            return left + right + 1;
        }
        return Math.min(left, right) + 1;
    }

    //[112].路径总和
    public boolean hasPathSum(TreeNode root, int targetSum) {
        //一定不是叶子节点
        if (root == null) return false;
        //判断叶子节点标准
        if (root.left == null && root.right == null) {
            return targetSum == root.val;
        }
        return hasPathSum(root.left, targetSum - root.val)
                || hasPathSum(root.right, targetSum - root.val);
    }

    //[113].路径总和 II
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> res = new ArrayList<>();
        dfsForPathSum(root, targetSum, new LinkedList<>(), res);
        return res;
    }

    private void dfsForPathSum(TreeNode root, int targetSum, LinkedList<Integer> path, List<List<Integer>> res) {
        if (root == null) return;

        //前序遍历
        path.addLast(root.val);
        if (root.left == null && root.right == null && targetSum == root.val) {
            res.add(new ArrayList<>(path));
            //回溯撤销节点的，加了return，会导致叶子节点会有撤销成功，导致路径上少减少一次撤销，从而使得下一次的选择会多一个节点。
            //主要取决于前序遍历顺序不能变更。
        }

        dfsForPathSum(root.left, targetSum - root.val, path, res);
        dfsForPathSum(root.right, targetSum - root.val, path, res);
        path.removeLast();
    }

    //[114].二叉树展开为链表
    public void flatten(TreeNode root) {
        if (root == null) return;

        flatten(root.left);
        flatten(root.right);

        TreeNode left = root.left;
        TreeNode right = root.right;

        root.right = left;
        root.left = null;

        TreeNode cur = root;
        //最后一个叶子节点
        while (cur.right != null) {
            cur = cur.right;
        }
        cur.right = right;
    }

    //[144].二叉树的前序遍历
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode cur = stack.pop();
            res.add(cur.val);

            if (cur.right != null) {
                stack.push(cur.right);
            }
            if (cur.left != null) {
                stack.push(cur.left);
            }
        }
        return res;
    }

    //[145].二叉树的后序遍历
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        while (cur != null || !stack.isEmpty()) {
            if (cur != null) {
                res.add(0, cur.val);
                stack.push(cur);
                cur = cur.right;
            } else {
                cur = stack.pop();
                cur = cur.left;
            }
        }
        return res;
    }

    //[116].填充每个节点的下一个右侧节点指针
    public Node connect(Node root) {
        if (root == null) return null;
        dfsForConnect(root.left, root.right);
        return root;
    }

    private void dfsForConnect(Node left, Node right) {
        if (left == null || right == null) {
            return;
        }
        left.next = right;
        dfsForConnect(left.left, left.right);
        dfsForConnect(right.left, right.right);
        dfsForConnect(left.right, right.left);
    }

    //[118].杨辉三角
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            List<Integer> level = new ArrayList<>();
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) {
                    level.add(1);
                } else {
                    level.add(res.get(i - 1).get(j - 1) + res.get(i - 1).get(j));
                }
            }
            res.add(level);
        }
        return res;
    }

    //[119].杨辉三角 II
    public List<Integer> getRow(int rowIndex) {
        List<Integer> res = new ArrayList<>();
        res.add(1);
        for (int i = 1; i <= rowIndex; i++) {
            res.add(0);
            for (int j = i; j > 0; j--) {
                res.set(j, res.get(j) + res.get(j - 1));
            }
        }
        return res;
    }

    //[120].三角形最小路径和
    public int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size();
        //走到(i, j)点的最小路径和
        int[][] dp = new int[n][n];
        dp[0][0] = triangle.get(0).get(0);
        for (int i = 1; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (j == 0) {
                    dp[i][j] = dp[i - 1][j] + triangle.get(i).get(j);
                } else if (j == i) {
                    dp[i][j] = dp[i - 1][j - 1] + triangle.get(i).get(j);
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle.get(i).get(j);
                }
            }
        }
        int min = dp[n - 1][0];
        for (int i = 1; i < n; i++) {
            min = Math.min(min, dp[n - 1][i]);
        }
        return min;
    }

    //[120].三角形最小路径和（空间压缩）
    public int minimumTotal2(List<List<Integer>> triangle) {
        int n = triangle.size();
        //到底层i的最短路径和
        int[] dp = new int[n];
        dp[0] = triangle.get(0).get(0);
        int pre = 0, cur;
        //  pre          cur, pre'     cur'
        // (i-1, j-1)   (i-1, j)     (i-1, j+1)
        //        ＼        ↓    ＼      ↓
        //               (i, j)       (i, j+1)
        for (int i = 1; i < n; i++) {
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
        int min = dp[0];
        for (int i = 1; i < n; i++) {
            min = Math.min(min, dp[i]);
        }
        return min;
    }

    //[121].买卖股票的最佳时机
    public int maxProfit(int[] prices) {
        int n = prices.length;
        //到第i天，0表示不持股票，1表示持有股票
        int[][] dp = new int[n][2];

        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for (int i = 1; i < n; i++) {
            //不持股票，（昨天卖掉股票的利润，今天卖掉股票的利润）
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);

            //持有股票，（昨天持有股票的利润， 今天购入股票的利润）
            dp[i][1] = Math.max(dp[i - 1][1], -prices[i]);
        }

        return dp[n - 1][0];
    }

    //[121].买卖股票的最佳时机 优化版
    public int maxProfit_y(int[] prices) {
        int n = prices.length;
        int dp_0 = 0;
        int dp_1 = -prices[0];
        for (int i = 1; i < n; i++) {
            //不持股票，（今天卖掉股票的利润，昨天卖掉股票的利润）
            dp_0 = Math.max(dp_0, dp_1 + prices[i]);
            //持有股票，（昨天持有股票的利润， 今天购入股票的利润）
            dp_1 = Math.max(dp_1, -prices[i]);
        }

        return dp_0;
    }

    //[122].买卖股票的最佳时机 II
    public int maxProfit2(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for (int i = 1; i < n; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }
        return dp[n - 1][0];
    }

    //[122].买卖股票的最佳时机 II 优化版
    public int maxProfit2_y(int[] prices) {
        int n = prices.length;
        int dp_0 = 0;
        int dp_1 = -prices[0];
        for (int i = 1; i < n; i++) {
            int pre_dp_0 = dp_0;
            int pre_dp_1 = dp_1;
            dp_0 = Math.max(pre_dp_0, pre_dp_1 + prices[i]);
            dp_1 = Math.max(pre_dp_1, pre_dp_0 - prices[i]);
        }
        return dp_0;
    }

    //[123].买卖股票的最佳时机 III
    public int maxProfit3(int[] prices) {
        // dp[i][k][0] = Math.max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
        // dp[i][k][1] = Math.max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
        int n = prices.length;
        int[][][] dp = new int[n][3][2];
        dp[0][1][0] = 0;
        dp[0][2][0] = 0;
        dp[0][1][1] = -prices[0];
        dp[0][2][1] = -prices[0];

        for (int i = 1; i < n; i++) {
            dp[i][2][0] = Math.max(dp[i - 1][2][0], dp[i - 1][2][1] + prices[i]);
            dp[i][2][1] = Math.max(dp[i - 1][2][1], dp[i - 1][1][0] - prices[i]);

            dp[i][1][0] = Math.max(dp[i - 1][1][0], dp[i - 1][1][1] + prices[i]);
            dp[i][1][1] = Math.max(dp[i - 1][1][1], -prices[i]);
        }
        return dp[n - 1][2][0];
    }

    //[123].买卖股票的最佳时机 III 优化版
    public int maxProfit3_y(int[] prices) {
        int n = prices.length;
        int dp_1_0 = 0;
        int dp_1_1 = -prices[0];
        int dp_2_0 = 0;
        int dp_2_1 = -prices[0];
        for (int i = 1; i < n; i++) {
            dp_2_0 = Math.max(dp_2_0, dp_2_1 + prices[i]);
            dp_2_1 = Math.max(dp_2_1, dp_1_0 - prices[i]);
            dp_1_0 = Math.max(dp_1_0, dp_1_1 + prices[i]);
            dp_1_1 = Math.max(dp_1_1, -prices[i]);
        }
        return dp_2_0;
    }

    //[128].最长连续序列
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int res = 0;
        for (int num : set) {
            if (!set.contains(num - 1)) {
                int cur = num;
                int longest = 1;
                while (set.contains(cur + 1)) {
                    cur += 1;
                    longest += 1;
                }
                res = Math.max(res, longest);
            }
        }
        return res;
    }

    //[129].求根节点到叶节点数字之和
    public int sumNumbers(TreeNode root) {
        return dfsForSumNumbers(root, 0);
    }

    private int dfsForSumNumbers(TreeNode root, int preVal) {
        if (root == null) return 0;
        int cur = preVal * 10 + root.val;
        if (root.left == null && root.right == null) {
            return cur;
        }
        return dfsForSumNumbers(root.right, cur) + dfsForSumNumbers(root.left, cur);
    }

    //[131].分割回文串
    public List<List<String>> partition(String s) {
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
        }
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                //三个字符，并且收尾相等，就是回文
                dp[i][j] = (dp[i + 1][j - 1] || j - i < 3) && s.charAt(i) == s.charAt(j);
            }
        }

        List<List<String>> res = new ArrayList<>();
        dfsForPartition(s, 0, res, new LinkedList<>(), dp);
        return res;
    }

    private void dfsForPartition(String s, int start, List<List<String>> res, LinkedList<String> path, boolean[][] dp) {
        if (start == s.length()) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = start; i < s.length(); i++) {
            if (!dp[start][i]) {
                continue;
            }
            path.addLast(s.substring(start, i + 1));
            dfsForPartition(s, i + 1, res, path, dp);
            path.removeLast();
        }
    }

    //[133].克隆图
    public Node cloneGraph(Node node) {
        if (node == null) return null;
        //hash赋值，同时充当访问标记
        Map<Node, Node> cloneMap = new HashMap<>();
        dfsForCloneGraph(node, cloneMap);
        return cloneMap.get(node);
    }

    private void dfsForCloneGraph(Node node, Map<Node, Node> cloneMap) {
        //已经创建过了，就不需要再次创建了
        if (cloneMap.containsKey(node)) {
            return;
        }
        Node clone = new Node(node.val);
        cloneMap.put(node, clone);
        if (node.neighbors != null && node.neighbors.size() > 0) {
            clone.neighbors = new ArrayList<>();
            for (Node neigh : node.neighbors) {
                dfsForCloneGraph(neigh, cloneMap);
                clone.neighbors.add(cloneMap.get(neigh));
            }
        }
    }

    //[134].加油站
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int left = 0, minLeft = Integer.MAX_VALUE;
        int n = gas.length;
        int start = 0;
        for (int i = 0; i < n; i++) {
            left += gas[i] - cost[i];
            //剩余最小，更新位置
            if (left < minLeft) {
                minLeft = left;
                start = i;
            }
        }
        //下一个位置才是起始位置，并且可能超过数组长度，并且是环
        return left >= 0 ? (start + 1) % n : -1;
    }

    //[136].只出现一次的数字
    public int singleNumber(int[] nums) {
        int res = nums[0];
        for (int i = 1; i < nums.length; i++) {
            res ^= nums[i];
        }
        return res;
    }

    //[137]只出现一次的数字 II
    public int singleNumber2(int[] nums) {
        int res = 0;
        for (int i = 0; i < 32; i++) {
            int count = 0;
            int pos = 1 << i;
            for (int num : nums) {
                if ((pos & num) == pos) {
                    count++;
                }
            }
            if (count % 3 != 0) {
                res |= 1 << i;
            }
        }
        return res;
    }

    //[19].删除链表的倒数第 N 个结点
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode slow = head, fast = head;
        int count = 0;
        while (count < n) {
            fast = fast.next;
            count++;
        }
        ListNode pre = null;
        while (fast != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next;
        }

        //第一个节点
        if (pre == null) {
            pre = slow.next;
            slow.next = null;
            return pre;
        } else {
            pre.next = slow.next;
            return head;
        }
    }

    //[21].合并两个有序链表
    public ListNode mergeTwoList(ListNode first, ListNode second) {
        ListNode dummyHead = new ListNode(-1), p = dummyHead;
        ListNode p1 = first, p2 = second;
        while (p1 != null && p2 != null) {
            if (p1.val < p2.val) {
                p.next = p1;
                p1 = p1.next;
            } else {
                p.next = p2;
                p2 = p2.next;
            }
            p = p.next;
        }

        if (p1 != null) {
            p.next = p1;
        }
        if (p2 != null) {
            p.next = p2;
        }
        return dummyHead.next;
    }

    //[23].合并K个升序链表
    public ListNode mergeKLists(ListNode[] lists) {
        //小顶堆
        Queue<ListNode> queue = new PriorityQueue<>((a, b) -> a.val - b.val);
        for (ListNode node : lists) {
            if (node != null) {
                queue.offer(node);
            }
        }
        ListNode dummyHead = new ListNode(-1), h = dummyHead;
        while (!queue.isEmpty()) {
            ListNode cur = queue.poll();
            h.next = cur;

            if (cur.next != null) {
                queue.offer(cur.next);
            }
            h = h.next;
        }
        return dummyHead.next;
    }

    //[24].两两交换链表中的节点
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode first = head, second = head.next;
        ListNode next = second.next;

        second.next = first;
        first.next = swapPairs(next);
        return second;
    }

    //[25].K个一组翻转链表
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null) return null;
        ListNode p = head;
        for (int i = 0; i < k; i++) {
            if (p == null) return head;
            p = p.next;
        }
        ListNode newHead = reverse(head, p);
        ListNode last = reverseKGroup(p.next, k);
        head.next = last;
        return newHead;
    }

    private ListNode reverse(ListNode head, ListNode end) {
        ListNode dummy = new ListNode(-1);
        ListNode p = head;
        while (p != end) {
            ListNode next = p.next;
            p.next = dummy.next;
            dummy.next = p;
            p = next;
        }

        end.next = dummy.next;
        dummy.next = end;
        return dummy.next;
    }

    //[61].旋转链表
    public ListNode rotateRight(ListNode head, int k) {
        int count = 0;
        ListNode p = head, last = null;
        while (p != null) {
            last = p;
            p = p.next;
            count++;
        }
        int realK;
        if (count == 0 || count == 1 || (realK = k % count) == 0) return head;

        ListNode slow = head, fast = head;
        for (int i = 0; i < realK; i++) {
            fast = fast.next;
        }
        ListNode pre = null;
        while (fast != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next;
        }
        pre.next = null;
        last.next = head;
        return slow;
    }

    //[82]删除排序链表中的重复元素 II
    public ListNode deleteDuplicates2(ListNode head) {
        ListNode slow = head, fast = head;
        return null;
    }

    //[83].删除排序链表中的重复元素
    public ListNode deleteDuplicates(ListNode head) {
        ListNode slow = head, fast = head;

        while (fast != null) {
            if (slow.val != fast.val) {
                slow.next = fast;
                slow = slow.next;
            }
            fast = fast.next;
        }
        slow.next = null;
        return head;
    }

    //[86].分隔链表
    public ListNode partition(ListNode head, int x) {
        ListNode sDummy = new ListNode(-1), fDummy = new ListNode(-1), cur = head, s = sDummy, f = fDummy;
        while (cur != null) {
            ListNode next = cur.next;
            if (cur.val < x) {
                s.next = cur;
                s = s.next;
            } else {
                f.next = cur;
                f = f.next;
            }
            //每次都断掉防止麻烦
            cur.next = null;
            cur = next;
        }
        s.next = fDummy.next;
        return sDummy.next;
    }

    //[92].反转链表 II
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode pre = dummy;
        for (int i = 0; i < left - 1; i++) {
            pre = pre.next;
        }
        ListNode cur = pre.next;
        ListNode next;
        for (int i = 0; i < right - left; i++) {
            next = cur.next;
            //移动的永远是cur
            cur.next = next.next;
            //此处的cur是会变动的，所以接的是pre.next节点
            next.next = pre.next;
            pre.next = next;
        }
        return dummy.next;
    }

    //[138].复制带随机指针的链表
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        //旧 + 新的
        Map<Node, Node> map = new HashMap<>();
        Node newHead = new Node(head.val);
        map.put(head, newHead);

        Node cur = head;
        while (cur != null) {
            Node copy = map.get(cur);

            if (cur.random != null) {
                map.putIfAbsent(cur.random, new Node(cur.random.val));
                copy.random = map.get(cur.random);
            }

            if (cur.next != null) {
                map.putIfAbsent(cur.next, new Node(cur.next.val));
                copy.next = map.get(cur.next);
            }

            cur = cur.next;
        }
        return newHead;
    }

    //[141].环形链表
    public boolean hasCycle(ListNode head) {
        ListNode slow = head, fast = head;

        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) return true;
        }
        return false;
    }

    //[142].环形链表 II
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) {
                break;
            }
        }
        if (fast == null || fast.next == null) return null;

        slow = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }

    //[143].重排链表
    public void reorderList(ListNode head) {
        if (head == null) return;
        ListNode slow = head, fast = head, mid = null;
        while (fast != null && fast.next != null) {
            mid = slow;
            fast = fast.next.next;
            slow = slow.next;
        }

        //奇数节点需要重置下
        if (fast != null) {
            mid = slow;
        }

        ListNode q = reverseList(mid.next);
        mid.next = null;
        ListNode p = head;
        while (q != null) {
            ListNode qNext = q.next;
            ListNode pNext = p.next;
            q.next = pNext;
            p.next = q;
            q = qNext;
            p = pNext;
        }
    }

    //[147].对链表进行插入排序
    public ListNode insertionSortList(ListNode head) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode cur = head.next, lastSorted = head;
        while (cur != null) {
            //维护最后排序好的指针
            if (lastSorted.val <= cur.val) {
                lastSorted = lastSorted.next;
            } else {
                //肯定要从头开始找
                ListNode pre = dummy;
                while (pre.next.val < cur.val) {
                    pre = pre.next;
                }
                //lastSorted后继节点指向cur后面的节点，因为cur之前都是排序好的
                lastSorted.next = cur.next;

                //pre的后面一个节点比较大，插入到pre后面
                cur.next = pre.next;
                pre.next = cur;
            }
            cur = lastSorted.next;
        }
        return dummy.next;
    }

    //[148].排序链表
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode fast = head, slow = head, pre = null;
        while (fast != null && fast.next != null) {
            pre = slow;
            fast = fast.next.next;
            slow = slow.next;
        }
        pre.next = null;

        //奇数情况，左边少一个， 右边多一个
        ListNode left = sortList(head);
        ListNode right = sortList(slow);
        return merge(left, right);
    }

    private ListNode merge(ListNode left, ListNode right) {
        ListNode dummy = new ListNode(-1), cur = dummy;
        while (left != null && right != null) {
            if (left.val < right.val) {
                cur.next = left;
                left = left.next;
            } else {
                cur.next = right;
                right = right.next;
            }
            cur = cur.next;
        }

        cur.next = left == null ? right : left;
        return dummy.next;
    }

    //[160].相交链表
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode p1 = headA, p2 = headB;
        while (p1 != p2) {
            if (p1 != null) {
                p1 = p1.next;
            } else {
                p1 = headB;
            }

            if (p2 != null) {
                p2 = p2.next;
            } else {
                p2 = headA;
            }
        }
        return p1;
    }

    //[203].移除链表元素
    public ListNode removeElements(ListNode head, int val) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode slow = dummy, fast = head;
        while (fast != null) {
            if (fast.val == val) {
                slow.next = fast.next;
            } else {
                slow = fast;
            }
            fast = fast.next;
        }
        return dummy.next;
    }

    //[206].反转链表
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode last = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return last;
    }

    //[206].反转链表 迭代
    public ListNode reverseList2(ListNode head) {
        ListNode cur = head, pre = null;
        while (cur != null) {
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }

    //[234].回文链表
    private ListNode left;

    public boolean isPalindrome(ListNode head) {
        left = head;
        return helperIsPalindrome(head);
    }

    private boolean helperIsPalindrome(ListNode right) {
        if (right == null) return true;
        boolean res = helperIsPalindrome(right.next);
        res = res && (left.val == right.val);
        left = left.next;
        return res;
    }

    //[237].删除链表中的节点
    public void deleteNode(ListNode node) {
        //不知道前面的节点，那只能去复制，删掉下一个节点
        node.val = node.next.val;
        node.next = node.next.next;
    }

    //[328].奇偶链表
    public ListNode oddEvenList(ListNode head) {
        ListNode oddHead = new ListNode(-1);
        ListNode evenHead = new ListNode(-1);
        ListNode cur = head, odd = oddHead, even = evenHead;
        for (int i = 1; cur != null; i++) {
            ListNode next = cur.next;
            if (i % 2 != 0) {
                odd.next = cur;
                odd = odd.next;
            } else {
                even.next = cur;
                even = even.next;
            }
            cur.next = null;
            cur = next;
        }
        odd.next = evenHead.next;
        return oddHead.next;
    }

    //[876].链表的中间结点
    public ListNode middleNode(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    //[198].打家劫舍
    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 0) return 0;
        if (n == 1) return nums[0];

        //dp[i] 从第i间房打劫的最大金额
        int[] dp = new int[n + 1];
        dp[n - 1] = nums[n - 1];
        //因为i需要从倒数第二个位置开始，所以需要n+1个空间
        for (int i = n - 2; i >= 0; i--) {
            dp[i] = Math.max(dp[i + 2] + nums[i], dp[i + 1]);
        }
        return dp[0];
    }

    //[213].打家劫舍 II
    public int rob2(int[] nums) {
        int len = nums.length;
        if (len == 0) return 0;
        if (len == 1) return nums[0];
        return Math.max(rob2(nums, 0, len - 2), rob2(nums, 1, len - 1));
    }

    private int rob2(int[] nums, int start, int end) {
        int dp_i = 0, dp_i_2 = 0, dp_i_1 = 0;
        for (int i = end; i >= start; i--) {
            dp_i = Math.max(dp_i_2 + nums[i], dp_i_1);
            dp_i_2 = dp_i_1;
            dp_i_1 = dp_i;
        }
        return dp_i;
    }

    //[337].打家劫舍 III
    private Map<TreeNode, Integer> map = new HashMap<>();

    public int rob(TreeNode root) {
        if (root == null) return 0;
        if (map.containsKey(root)) return map.get(root);
        int rob_it = root.val +
                (root.left != null ? rob(root.left.left) + rob(root.left.right) : 0) +
                (root.right != null ? rob(root.right.left) + rob(root.right.right) : 0);
        int rob_not = rob(root.left) + rob(root.right);
        int res = Math.max(rob_it, rob_not);
        map.put(root, res);
        return res;
    }

    //[146].LRU 缓存机制
    class LRUCache {

        public LRUCache(int capacity) {

        }

        public int get(int key) {
            return 0;
        }

        public void put(int key, int value) {

        }
    }

    //[150].逆波兰表达式求值
    public int evalRPN(String[] tokens) {
        Stack<String> stack = new Stack<>();
        for (String token : tokens) {
            if (token.equals("+")
                    || token.equals("-")
                    || token.equals("*")
                    || token.equals("/")) {
                int second = Integer.parseInt(stack.pop());
                int first = Integer.parseInt(stack.pop());
                if (token.equals("+")) {
                    stack.push("" + (second + first));
                } else if (token.equals("-")) {
                    stack.push("" + (first - second));
                } else if (token.equals("*")) {
                    stack.push("" + (first * second));
                } else {
                    stack.push("" + (first / second));
                }
            } else {
                stack.push(token);
            }
        }
        return stack.isEmpty() ? 0 : Integer.parseInt(stack.pop());
    }

    //[151].翻转字符串里的单词
    public String reverseWords(String s) {
        StringBuilder sb = new StringBuilder();
        int len = s.length();
        int left = 0;
        while (s.charAt(left) == ' ') {
            left++;
        }

        for (int i = len - 1; i >= left; i--) {
            int j = i;
            while (i >= left && s.charAt(i) != ' ') {
                i--;
            }

            if (i != j) {
                sb.append(s.substring(i + 1, j + 1));
                if (i > left) {
                    sb.append(" ");
                }
            }
        }
        return sb.toString();
    }

    //[33].搜索旋转排序数组
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            }

            //说明left到mid是严格递增
            if (nums[mid] > nums[right]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    //mid肯定不等于target
                    left = mid + 1;
                }
            } else {
                //说明mid到right是严格递增
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    //mid肯定不等于target
                    right = mid - 1;
                }
            }
        }
        return -1;
    }

    //[34].在排序数组中查找元素的第一个和最后一个位置
    public int[] searchRange(int[] nums, int target) {
        int leftBound = findIndex(nums, target, true);
        int rightBound = findIndex(nums, target, false);
        return new int[]{leftBound, rightBound};
    }

    private int findIndex(int[] nums, int target, boolean isLeft) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                if (isLeft) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (isLeft) {
            if (left >= nums.length || nums[left] != target) return -1;
            return left;
        } else {
            if (right < 0 || nums[right] != target) return -1;
            return right;
        }
    }

    private int findLeftIndex(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (target == nums[mid]) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else if (target < nums[mid]) {
                right = mid - 1;
            }
        }

        if (left >= nums.length || nums[left] != target) return -1;
        return left;
    }

    private int findRightIndex(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (target == nums[mid]) {
                left = mid + 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else if (target < nums[mid]) {
                right = mid - 1;
            }
        }

        if (right < 0 || nums[right] != target) return -1;
        return right;
    }

    //[35].搜索插入位置
    public int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }

    //[74].搜索二维矩阵
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int left = 0, right = m * n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int x = mid / m;
            int y = mid % m;
            if (matrix[x][y] == target) {
                return true;
            } else if (matrix[x][y] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return false;
    }

    //[81].搜索旋转排序数组 II
    public boolean search3(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] == target) {
                return true;
            }

            //10111 11101 这两种情况中，没办法判断递增区间走向，所以砍掉左边的相同的
            if (nums[mid] == nums[left]) {
                left++;
            } else if (nums[mid] > nums[right]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return false;
    }

    //[240].搜索二维矩阵 II
    public boolean searchMatrix2(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int x = 0, y = n - 1;

        while (x >= 0 && x < m && y >= 0 && y < n) {
            if (matrix[x][y] == target) {
                return true;
            } else if (matrix[x][y] < target) {
                x++;
            } else {
                y--;
            }
        }
        return false;
    }

    //[153].寻找旋转排序数组中的最小值
    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return nums[left];
    }

    //[162].寻找峰值
    public int findPeakElement(int[] nums) {
        int left = 0, right = nums.length - 1;
        //因为需要判断后一个值，所以此处是left < right
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    //[37].解数独
    public void solveSudoku(char[][] board) {
        backtraceSolveSudoku(board, 0, 0);
    }

    private boolean backtraceSolveSudoku(char[][] board, int x, int y) {
        if (y == 9) {
            return backtraceSolveSudoku(board, x + 1, 0);
        }

        //已经第10行了
        if (x == 9) {
            return true;
        }

        if (board[x][y] != '.') {
            return backtraceSolveSudoku(board, x, y + 1);
        }
        for (char i = '1'; i <= '9'; i++) {
            if (!isValidSudoku(board, x, y, i)) {
                continue;
            }
            board[x][y] = i;
            if (backtraceSolveSudoku(board, x, y + 1)) return true;
            board[x][y] = '.';
        }
        return false;
    }

    private boolean isValidSudoku(char[][] board, int x, int y, char ch) {
        for (int i = 0; i < 9; i++) {
            if (board[x][i] == ch) return false;
            if (board[i][y] == ch) return false;
            if (board[(x / 3) * 3 + i / 3][(y / 3) * 3 + i % 3] == ch) return false;
        }
        return true;
    }

    //[17].电话号码的字母组合
    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        if (digits.length() == 0) return res;
        String[] numbers = new String[]{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        backtraceForLetterCombinations(digits, 0, res, new StringBuilder(), numbers);
        return res;
    }

    private void backtraceForLetterCombinations(String digits, int s, List<String> res, StringBuilder sb, String[] numbers) {
        if (s == digits.length()) {
            res.add(sb.toString());
            return;
        }
        char ch = digits.charAt(s);
        for (char choice : numbers[ch - '0'].toCharArray()) {
            sb.append(choice);
            backtraceForLetterCombinations(digits, s + 1, res, sb, numbers);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    //[39].组合总和
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (candidates.length == 0) return res;
        backtraceForCombinationSum(candidates, 0, target, res, new LinkedList<>());
        return res;
    }

    private void backtraceForCombinationSum(int[] candidates, int s, int target, List<List<Integer>> res, LinkedList<Integer> select) {
        if (target == 0) {
            res.add(new ArrayList<>(select));
            return;
        }

        for (int i = s; i < candidates.length; i++) {
            if (candidates[i] > target) {
                continue;
            }
            select.addLast(candidates[i]);
            backtraceForCombinationSum(candidates, i, target - candidates[i], res, select);
            select.removeLast();
        }
    }

    //[40]组合总和 II
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (candidates.length == 0) return res;
        Arrays.sort(candidates);
        backtraceForCombinationSum2(candidates, 0, target, res, new LinkedList<>());
        return res;
    }

    private void backtraceForCombinationSum2(int[] candidates, int s, int target, List<List<Integer>> res, LinkedList<Integer> select) {
        if (target == 0) {
            res.add(new ArrayList<>(select));
            return;
        }

        for (int i = s; i < candidates.length; i++) {
            //本层如果有重复的话，只选第一个，后面的直接跳过
            if (i > s && candidates[i] == candidates[i - 1]) {
                continue;
            }
            if (candidates[i] > target) {
                continue;
            }
            select.addLast(candidates[i]);
            //元素不能重复选择，只能从下一个选择
            backtraceForCombinationSum2(candidates, i + 1, target - candidates[i], res, select);
            select.removeLast();
        }
    }

    //[216].组合总和 III
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> res = new ArrayList<>();
        if (n < 0) return res;
        backtraceForCombinationSum3(k, n, 1, res, new LinkedList<>());
        return res;
    }

    private void backtraceForCombinationSum3(int k, int n, int s, List<List<Integer>> res, LinkedList<Integer> select) {
        if (select.size() == k) {
            if (n == 0) {
                res.add(new ArrayList<>(select));
            }
            return;
        }

        for (int i = s; i <= 9; i++) {
            if (i > n) {
                continue;
            }
            select.addLast(i);
            backtraceForCombinationSum3(k, n - i, i + 1, res, select);
            select.removeLast();
        }
    }

    //[51].N皇后
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            sb.append('.');
        }
        LinkedList select = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            select.add(sb.toString());
        }
        backtraceForSolveNQueens(n, 0, res, select);
        return res;
    }

    private void backtraceForSolveNQueens(int n, int row, List<List<String>> res, LinkedList<String> select) {
        if (row == n) {
            res.add(new ArrayList<>(select));
            return;
        }
        char[] arr = select.get(row).toCharArray();
        for (int col = 0; col < n; col++) {
            if (!isValid(select, row, col)) {
                continue;
            }
            arr[col] = 'Q';
            select.set(row, new String(arr));

            backtraceForSolveNQueens(n, row + 1, res, select);

            arr[col] = '.';
            select.set(row, new String(arr));
        }
    }

    private boolean isValid(LinkedList<String> select, int row, int col) {
        int n = select.get(row).length();
        //每次行上是一个选择，所以不需要判断行是否合法，因为肯定合法
        //列不合法
        for (int i = 0; i < n; i++) {
            if (select.get(i).charAt(col) == 'Q') {
                return false;
            }
        }
        //左上不合法
        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
            if (select.get(i).charAt(j) == 'Q') {
                return false;
            }
        }

        //右上不合法
        for (int i = row, j = col; i >= 0 && j < n; i--, j++) {
            if (select.get(i).charAt(j) == 'Q') {
                return false;
            }
        }
        return true;
    }

    //[77].组合
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        backtraceForCombine(n, k, 1, res, new LinkedList<>());
        return res;
    }

    private void backtraceForCombine(int n, int k, int s, List<List<Integer>> res, LinkedList<Integer> select) {
        if (select.size() == k) {
            res.add(new ArrayList<>(select));
            return;
        }

        for (int i = s; i <= n; i++) {
            select.addLast(i);
            backtraceForCombine(n, k, i + 1, res, select);
            select.removeLast();
        }
    }

    //[78].子集
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        backtraceForSubsets(nums, 0, res, new LinkedList<>());
        return res;
    }

    private void backtraceForSubsets(int[] nums, int s, List<List<Integer>> res, LinkedList<Integer> select) {
        //1 2 3
        res.add(new ArrayList<>(select));
        for (int i = s; i < nums.length; i++) {
            select.addLast(nums[i]);
            backtraceForSubsets(nums, i + 1, res, select);
            select.removeLast();
        }
    }

    //[90].子集 II
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        backtraceForSubsetsWithDup(nums, 0, res, new LinkedList<>());
        return res;
    }

    private void backtraceForSubsetsWithDup(int[] nums, int s, List<List<Integer>> res, LinkedList<Integer> select) {
        res.add(new ArrayList<>(select));
        for (int i = s; i < nums.length; i++) {
            //本层中有重复的，则跳过，永远直选第一个
            if (i > s && nums[i] == nums[i - 1]) {
                continue;
            }

            select.addLast(nums[i]);
            backtraceForSubsetsWithDup(nums, i + 1, res, select);
            select.removeLast();
        }
    }

    //[79].单词搜索
    public boolean exist(char[][] board, String word) {
        int m = board.length;
        int n = board[0].length;
        boolean[][] visit = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (backtraceForExist(board, i, j, word, 0, visit)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean backtraceForExist(char[][] board, int x, int y, String word, int s, boolean[][] visit) {
        if (board[x][y] != word.charAt(s)) {
            return false;
        }
        if (s == word.length() - 1) {
            return true;
        }

        visit[x][y] = true;
        int[][] direct = new int[][]{{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
        int m = board.length;
        int n = board[0].length;
        for (int[] di : direct) {
            int newX = x + di[0];
            int newY = y + di[1];
            if (newX >= 0 && newX < m
                    && newY >= 0 && newY < n
                    && !visit[newX][newY]
                    && backtraceForExist(board, newX, newY, word, s + 1, visit)) {
                return true;
            }
        }

        visit[x][y] = false;
        return false;
    }

    //[89].格雷编码
    public List<Integer> grayCode(int n) {
        boolean[] visit = new boolean[1 << n];
        LinkedList<Integer> select = new LinkedList<>();
        backtraceForGrayCode(n, 0, visit, select);
        return select;
    }

    private boolean backtraceForGrayCode(int n, int cur, boolean[] visit, LinkedList<Integer> select) {
        if (select.size() == 1 << n) {
            return true;
        }

        select.add(cur);
        visit[cur] = true;
        for (int i = 0; i < n; i++) {
            int next = cur ^ 1 << i;
            if (!visit[next] && backtraceForGrayCode(n, next, visit, select)) {
                return true;
            }
        }
        visit[cur] = false;
        return false;
    }

    //[42].接雨水
    public int trap(int[] height) {
        int res = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < height.length; i++) {
            int curHeight = height[i];
            int maxHeight = 0;
            while (!stack.isEmpty() && curHeight >= height[stack.peek()]) {
                int top = stack.pop();
                if (stack.isEmpty()) {
                    continue;
                }
                maxHeight = Math.max(maxHeight, height[top]);
                int left = stack.peek();
                int width = i - left - 1;

                res += width * (Math.min(height[left], curHeight) - maxHeight);
            }
            stack.push(i);
        }
        return res;
    }

    //[456].132 模式
    public boolean find132pattern(int[] nums) {
        int n = nums.length;
        if (n <= 2) return false;
        //[3,5,0,3,4]
        Stack<Integer> stack = new Stack<>();
        //维护是一个单调递减栈
        int maxRight = Integer.MIN_VALUE;
        for (int i = n - 1; i >= 0; i--) {
            if (nums[i] < maxRight) {
                return true;
            }
            while (!stack.isEmpty() && nums[i] > stack.peek()) {
                maxRight = Math.max(maxRight, stack.pop());
            }
            stack.push(nums[i]);
        }
        return false;
    }

    //[496].下一个更大元素I
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Map<Integer, Integer> numberMap = new HashMap<>();
        Stack<Integer> stack = new Stack<>();
        for (int i = nums2.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && nums2[i] >= nums2[stack.peek()]) {
                stack.pop();
            }
            numberMap.put(nums2[i], stack.isEmpty() ? -1 : nums2[stack.peek()]);
            stack.push(i);
        }

        int[] res = new int[nums1.length];
        for (int i = 0; i < nums1.length; i++) {
            res[i] = numberMap.get(nums1[i]);
        }
        return res;
    }

    //[383].赎金信
    public boolean canConstruct(String ransomNote, String magazine) {
        if (ransomNote.length() > magazine.length()) return false;
        int[] count = new int[26];
        for (int i = 0; i < magazine.length(); i++) {
            count[magazine.charAt(i) - 'a']++;
        }
        for (int i = 0; i < ransomNote.length(); i++) {
            if (--count[ransomNote.charAt(i) - 'a'] < 0) {
                return false;
            }
        }
        return true;
    }

    //[372].超级次方
    public int superPow(int a, int[] b) {
        return dfsForSuperPow(a, b, b.length - 1);
    }

    private int dfsForSuperPow(int a, int[] b, int index) {
        if (index < 0) {
            return 1;
        }
        int base = 1337;
        int part1 = myPow(a, b[index]);
        int part2 = myPow(dfsForSuperPow(a, b, index - 1), 10);
        return part1 * part2 % base;
    }

    //快速幂
    private int myPow(int a, int b) {
        if (b == 0) return 1;
        int base = 1337;
        a %= base;

        if (b % 2 == 0) {
            int half = myPow(a, b / 2);
            return half * half % base;
        } else {
            return a * myPow(a, b - 1) % base;
        }
    }

    //[50].Pow(x, n)
    public double myPow(double x, int n) {
        if (n == 0) return 1;
        if (n == 1) return x;
        if (n == -1) return 1 / x;

        double half = myPow(x, n / 2);
        double left = myPow(x, n % 2);
        return half * half * left;
    }

    //[581].最短无序连续子数组
    public int findUnsortedSubarray(int[] nums) {
        int right = 0;
        int left = nums.length - 1;
        Stack<Integer> stack = new Stack<>();
        for (int i = nums.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && nums[i] > nums[stack.peek()]) {
                right = Math.max(right, stack.pop());
            }
            stack.push(i);
        }

        //一个栈没办法同时找到左右边界
        stack.clear();
        for (int i = 0; i < nums.length; i++) {
            while (!stack.isEmpty() && nums[i] < nums[stack.peek()]) {
                left = Math.min(left, stack.pop());
            }
            stack.push(i);
        }

        return right > left ? right - left + 1 : 0;
    }

    //[739].每日温度
    public int[] dailyTemperatures(int[] temperatures) {
        int[] res = new int[temperatures.length];
        Stack<Integer> stack = new Stack<>();
        //单调递增减栈，遇到小的压掉它
        for (int i = temperatures.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                stack.pop();
            }

            res[i] = stack.isEmpty() ? 0 : stack.peek() - i;
            stack.push(i);
        }
        return res;
    }

    //[503].下一个更大的元素II
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        Stack<Integer> stack = new Stack<>();
        for (int i = 2 * n - 1; i >= 0; i--) {
            while (!stack.isEmpty() && nums[i % n] >= stack.peek()) {
                stack.pop();
            }

            res[i % n] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(nums[i % n]);
        }
        return res;
    }

    //[300].最长递增子序列
    public int lengthOfLIS(int[] nums) {
        int size = nums.length;
        if (size == 0) return 0;
        int[] dp = new int[size];
        int length = 1;
        for (int i = 0; i < size; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            length = Math.max(length, dp[i]);
        }

        return length;
    }

    //[354].俄罗斯套娃信封问题
    public int maxEnvelopes(int[][] envelopes) {
        int n = envelopes.length;
        Arrays.sort(envelopes, (a, b) -> {
            if (a[0] == b[0]) return b[1] - a[1];
            else return a[0] - b[0];
        });
        int[] height = new int[n];
        for (int i = 0; i < n; i++) {
            height[i] = envelopes[i][1];
        }
        return lengthOfLIS(height);
    }

    //[392].判断子序列
    public boolean isSubsequence(String s, String t) {
        int i = 0, j = 0;
        while (i < s.length() && j < t.length()) {
            if (s.charAt(i) == t.charAt(j)) {
                i++;
                j++;
            }
            j++;
        }
        return i == s.length();
    }

    //[172].阶乘后的零
    public int trailingZeroes(int n) {
        int res = 0;
        long d = 5;
        while (n >= d) {
            res += n / d;
            d *= 5;
        }
        return res;
    }

    //[199].二叉树的右视图
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        dfsForRightSideView(root, 0, res);
        return res;
    }

    private void dfsForRightSideView(TreeNode root, int depth, List<Integer> res) {
        if (root == null) return;
        if (res.size() == depth) {
            res.add(root.val);
        }
        dfsForRightSideView(root.right, depth + 1, res);
        dfsForRightSideView(root.left, depth + 1, res);
    }

    //[201].数字范围按位与
    public int rangeBitwiseAnd(int left, int right) {
        int m = left, n = right;
        int count = 0;
        while (m < n) {
            count++;
            m >>= 1;
            n >>= 1;
        }
        return m << count;
    }

    //[1816].截断句子
    public String truncateSentence(String s, int k) {
        int count = 0, end = 0;
        for (int i = 1; i <= s.length(); i++) {
            if (i == s.length() || s.charAt(i) == ' ') {
                count++;
                if (count == k) {
                    end = i;
                    break;
                }
            }
        }
        return s.substring(0, end);
    }

    //[491].递增子序列
    public List<List<Integer>> findSubsequences(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        backtraceForFindSubsequences(nums, 0, res, new LinkedList<>());
        return res;
    }

    private void backtraceForFindSubsequences(int[] nums, int s, List<List<Integer>> res, LinkedList<Integer> select) {
        if (select.size() >= 2) {
            res.add(new ArrayList<>(select));
        }

        Set<Integer> set = new HashSet<>();
        for (int i = s; i < nums.length; i++) {
            if (set.contains(nums[i])) {
                continue;
            }
            if (select.size() > 0 && nums[i] < select.getLast()) {
                continue;
            }

            set.add(nums[i]);
            select.addLast(nums[i]);
            backtraceForFindSubsequences(nums, i + 1, res, select);
            select.removeLast();
        }
    }

    //[209].长度最小的子数组
    public int minSubArrayLen(int target, int[] nums) {
        //2,3,1,2,4, 3
        //2 5 6 8 12 15
        int n = nums.length;
        int[] preSum = new int[n + 1];
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += nums[i];
            preSum[i + 1] = sum;
        }
        int left = 0, right = 0, res = Integer.MAX_VALUE;
        while (right < nums.length) {
            right++;
            //找到合理的，就缩
            while (preSum[right] - preSum[left] >= target) {
                res = Math.min(res, right - left);
                left++;
            }

        }
        return res == Integer.MAX_VALUE ? 0 : res;
    }

    //[674].最长连续递增序列
    public static int findLengthOfLCIS(int[] nums) {
        //1 2 5 4 7
        int n = nums.length;
        if (n == 1) return 1;
        int left = 0, right = 1, res = 0;
        while (right < n) {
            if (nums[right - 1] >= nums[right]) {
                left = right;
            }
            res = Math.max(res, right - left + 1);
            right++;
        }
        return res;
    }

    //[56].合并区间
    public int[][] merge(int[][] intervals) {
        //1 2  1 5  3 6  7 9
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        LinkedList<int[]> res = new LinkedList<>();
        res.addLast(intervals[0]);
        for (int i = 1; i < intervals.length; i++) {
            int[] last = res.getLast();
            if (intervals[i][0] <= last[1]) {
                last[1] = Math.max(last[1], intervals[i][1]);
            } else {
                res.addLast(intervals[i]);
            }
        }
        return res.toArray(new int[res.size()][]);
    }

    //[57].插入区间
    public int[][] insert(int[][] intervals, int[] newInterval) {
        if (newInterval.length == 0) return intervals;
        LinkedList<int[]> res = new LinkedList<>();
        int left = newInterval[0], right = newInterval[1];
        for (int i = 0; i < intervals.length; i++) {
            int[] interval = intervals[i];
            if (interval[0] > right) {
                res.add(new int[]{left, right});
                //在插入区间右侧，且不相交，则添加插入区域
                //同时更新插入区域，保证下一次添加为插入区域
                left = interval[0];
                right = interval[1];
            } else if (left > interval[1]) {
                //在插入区间左侧，且不相交，则添加左侧区域
                res.add(interval);
            } else {
                //相交，维护区域
                left = Math.min(interval[0], left);
                right = Math.max(interval[1], right);
            }
        }
        //还剩最后一个区域需要添加
        res.add(new int[]{left, right});
        return res.toArray(new int[][]{});
    }

    //[436].
    public int[] findRightInterval(int[][] intervals) {
        return null;
    }

    //[986].区间列表的交集
    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
        return null;
    }

    //[1288].删除被覆盖的区间
    public int removeCoveredIntervals(int[][] intervals) {
        return 0;
    }

    //[689].三个无重叠子数组的最大和
    public int[] maxSumOfThreeSubarrays(int[] nums, int k) {
        int[] res = new int[3];
        int sum1 = 0, maxSum1 = 0, maxIdx1 = 0;
        int sum2 = 0, maxSum12 = 0, maxIdx21 = 0, maxIdx22 = 0;
        int sum3 = 0, maxSum123 = 0;

        for (int i = 2 * k; i < nums.length; i++) {
            sum1 += nums[i - 2 * k];
            sum2 += nums[i - k];
            sum3 += nums[i];
            if (i >= 3 * k - 1) {
                if (sum1 > maxSum1) {
                    maxSum1 = sum1;
                    maxIdx1 = i - 3 * k + 1;
                }
                if (sum2 + maxSum1 > maxSum12) {
                    maxSum12 = sum2 + maxSum1;
                    maxIdx21 = maxIdx1;
                    maxIdx22 = i - 2 * k + 1;
                }
                if (sum3 + maxSum12 > maxSum123) {
                    maxSum123 = sum3 + maxSum12;
                    res[0] = maxIdx21;
                    res[1] = maxIdx22;
                    res[2] = i - k + 1;
                }

                sum1 -= nums[i - 3 * k + 1];
                sum2 -= nums[i - 2 * k + 1];
                sum3 -= nums[i - k + 1];
            }
        }
        return res;
    }

    //[435].无重叠区间
    public int eraseOverlapIntervals(int[][] intervals) {
        int n = intervals.length;
        int overlap = 0;
        //1,2  2,3  3,4  1,3
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        int end = intervals[0][1];
        for (int i = 1; i < n; i++) {
            int[] cur = intervals[i];
            if (cur[0] < end) {
                overlap++;
                //默认保留最短的，长的覆盖范围更大，所以删除
                end = Math.min(end, cur[1]);
            } else {
                end = cur[1];
            }
        }
        return overlap;
    }

    //[215].数组中的第K个最大元素
    public int findKthLargest(int[] nums, int k) {
        //第K大意味着是从小到大是第n-k位
        return quickSort(nums, 0, nums.length - 1, nums.length - k);
    }

    private int quickSort(int[] nums, int left, int right, int targetIndex) {
        int l = left, r = right;
        int pivot = nums[l];
        while (l < r) {
            while (l < r && pivot <= nums[r]) r--;
            nums[l] = nums[r];

            while (l < r && nums[l] <= pivot) l++;
            nums[r] = nums[l];
        }
        nums[l] = pivot;

        if (l == targetIndex) {
            return nums[l];
        } else if (l < targetIndex) {
            return quickSort(nums, l + 1, right, targetIndex);
        } else {
            return quickSort(nums, left, l - 1, targetIndex);
        }
    }

    //[221].最大正方形
    public int maximalSquare(char[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        //以i,j为终点的能构成正方形的最大边长
        int[][] dp = new int[m][n];
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
                    }
                    res = Math.max(dp[i][j], res);
                }
            }
        }
        return res * res;
    }

    //[279].完全平方数
    public int numSquares(int n) {
        if (n <= 0) return 0;
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            dp[i] = Integer.MAX_VALUE;
            for (int j = 1; j * j <= i; j++) {
                dp[i] = Math.min(dp[i - j * j] + 1, dp[i]);
            }
        }
        return dp[n];
    }


    public static void main(String[] args) {
        System.out.println(new AllOfThem().permute(new int[]{1, 2, 3}));
        System.out.println(new AllOfThem().permuteUnique(new int[]{1, 1, 2}));

        System.out.println(new AllOfThem().findMinHeightTrees(2, new int[][]{{0, 1}}));
        System.out.println(new AllOfThem().findMinHeightTrees(6, new int[][]{{3, 0}, {3, 1}, {3, 2}, {3, 4}, {5, 4}}));
        System.out.println(new AllOfThem().numTrees(3));
        System.out.println(new AllOfThem().generate(5));
        System.out.println(new AllOfThem().getRow(4));
        System.out.println(new AllOfThem().partition("aabc"));
        System.out.println(new AllOfThem().singleNumber(new int[]{1, 2, 1}));
        System.out.println(new AllOfThem().singleNumber(new int[]{4, 1, 2, 1, 2}));
        System.out.println(new AllOfThem().singleNumber2(new int[]{0, 1, 0, 1, 0, 1, -99}));
        ListNode list = new ListNode(1);
        list.next = new ListNode(2);
        list.next.next = new ListNode(3);
        list.next.next.next = new ListNode(4);
        ListNode r1 = new AllOfThem().swapPairs(list);

        ListNode second = new ListNode(1);
        second.next = new ListNode(2);
        second.next.next = new ListNode(3);
        ListNode result = new AllOfThem().mergeTwoList(list, second);

        ListNode l61 = new ListNode(1);
        l61.next = new ListNode(2);
        l61.next.next = new ListNode(3);
        l61.next.next.next = new ListNode(4);
        l61.next.next.next.next = new ListNode(5);
        ListNode r61 = new AllOfThem().rotateRight(l61, 2);

        ListNode l141 = new ListNode(1);
        l141.next = new ListNode(2);
        l141.next.next = new ListNode(3);
        l141.next.next.next = new ListNode(4);
//        l141.next.next.next.next = new ListNode(5);
        new AllOfThem().reorderList(l141);


        ListNode l328 = new ListNode(1);
        l328.next = new ListNode(2);
        l328.next.next = new ListNode(3);
        l328.next.next.next = new ListNode(4);
        ListNode r328 = new AllOfThem().oddEvenList(l328);

        System.out.println(new AllOfThem().reverseWords("  Bob    Loves  Alice   "));

        System.out.println(new AllOfThem().findLeftIndex(new int[]{1}, 5));
        System.out.println(new AllOfThem().searchInsert(new int[]{1}, 5));
        System.out.println(new AllOfThem().search3(new int[]{1, 0, 1, 1, 1}, 0));

        System.out.println(new AllOfThem().letterCombinations("23"));
        System.out.println(new AllOfThem().combinationSum(new int[]{2, 3, 4, 7}, 7));
        System.out.println(new AllOfThem().combinationSum2(new int[]{10, 1, 2, 7, 6, 1, 5}, 8));
        System.out.println(new AllOfThem().combinationSum3(3, 9));
        System.out.println(new AllOfThem().subsets(new int[]{1, 2, 3, 4}));
        System.out.println(new AllOfThem().subsetsWithDup(new int[]{1, 2, 2}));
        System.out.println(new AllOfThem().trap(new int[]{0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1}));
        System.out.println(new AllOfThem().trap(new int[]{4, 2, 0, 3, 2, 5}));
        System.out.println(Arrays.toString(new AllOfThem().nextGreaterElement(new int[]{4, 1, 2}, new int[]{1, 3, 4, 2})));
        System.out.println(new AllOfThem().combine(4, 2));
        System.out.println(new AllOfThem().solveNQueens(4));
//        System.out.println(new AllOfThem().find132pattern(new int[]{1,2,3,4}));
//        System.out.println(new AllOfThem().find132pattern(new int[]{3,1,4,2}));
        System.out.println(new AllOfThem().find132pattern(new int[]{-1, 3, 2, 0}));
//        System.out.println(new AllOfThem().find132pattern(new int[]{3, 5, 0, 3, 4}));
        System.out.println(Arrays.toString(new AllOfThem().dailyTemperatures(new int[]{30, 60, 90})));
        System.out.println(Arrays.toString(new AllOfThem().dailyTemperatures(new int[]{73, 74, 75, 71, 69, 72, 76, 73})));
        System.out.println(Arrays.toString(new AllOfThem().dailyTemperatures(new int[]{73})));
        System.out.println(new AllOfThem().findUnsortedSubarray(new int[]{2, 6, 4, 8, 10, 9, 15}));
        System.out.println(new AllOfThem().findUnsortedSubarray(new int[]{1, 2, 3, 4}));
        System.out.println(new AllOfThem().findUnsortedSubarray(new int[]{1, 2, 3, 3, 3}));
        System.out.println(new AllOfThem().findUnsortedSubarray(new int[]{1}));
        System.out.println(new AllOfThem().grayCode(3));
        System.out.println(new AllOfThem().superPow(2, new int[]{1, 0}));
        System.out.println(new AllOfThem().trailingZeroes(10000));

        TreeNode t199 = new TreeNode(1);
        t199.left = new TreeNode(2);
        t199.right = new TreeNode(3);
        t199.left.right = new TreeNode(5);
        t199.right.left = new TreeNode(4);
        System.out.println(new AllOfThem().rightSideView(t199));
        System.out.println(new AllOfThem().merge(new int[][]{{1, 4}, {1, 4}}));

        System.out.println(new AllOfThem().numSquares(12));
    }
}
