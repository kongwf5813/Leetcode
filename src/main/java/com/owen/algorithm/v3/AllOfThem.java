package com.owen.algorithm.v3;

import java.util.*;

public class AllOfThem {
    public class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
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


    public static void main(String[] args) {
        System.out.println(new AllOfThem().permute(new int[]{1, 2, 3}));
        System.out.println(new AllOfThem().permuteUnique(new int[]{1, 1, 2}));

        System.out.println(new AllOfThem().findMinHeightTrees(2, new int[][]{{0, 1}}));
        System.out.println(new AllOfThem().findMinHeightTrees(6, new int[][]{{3, 0}, {3, 1}, {3, 2}, {3, 4}, {5, 4}}));
    }
}
