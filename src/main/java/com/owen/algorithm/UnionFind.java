package com.owen.algorithm;

public class UnionFind {
    public static class UF {

        private int[] parent;
        private int[] size;

        public UF(int count) {
            parent = new int[count];
            size = new int[count];
            for (int i = 0; i < count; i++) {
                size[i] = 1;
                parent[i] = i;
            }
        }

        public void union(int p, int q) {
            int rootP = find(p);
            int rootQ = find(q);

            if (size[rootP] > size[rootQ]) {
                parent[rootQ] = rootP;
                size[rootP] += size[rootQ];
            } else {
                parent[rootP] = rootQ;
                size[rootQ] += size[rootP];
            }
        }

        public int find(int x) {
            while (x != parent[x]) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }

        public int getParent(int i) {
            return parent[i];
        }

        public boolean connected(int p, int q) {
            int rootP = find(p);
            int rootQ = find(q);
            return rootP == rootQ;
        }
    }

    //[547].省份数量
    public static int findCircleNum(int[][] isConnected) {
        UF uf = new UF(isConnected.length);
        for (int i = 0; i < isConnected.length; i++) {
            for (int j = 0; j < isConnected[0].length; j++) {
                if (isConnected[i][j] == 1) {
                    uf.union(i, j);
                }
            }
        }
        int res = 0;
        for (int i = 0; i < isConnected.length; i++) {
            if (i == uf.getParent(i)) {
                res++;
            }
        }
        return res;
    }

    //[684].冗余连接
    public static int[] findRedundantConnection(int[][] edges) {
        int nodeCount = edges.length;
        //边是从1开始记
        UF uf = new UF(nodeCount + 1);
        for (int i = 0; i < nodeCount; i++) {
            int[] edge = edges[i];
            //刚开始不联通，然后联通
            if (!uf.connected(edge[0], edge[1])) {
                uf.union(edge[0], edge[1]);
            } else {
                //居然联通了，那么这条边就是多余的边
                return edge;
            }
        }
        return new int[0];
    }

    //[990].等式方程的可满足性
    public static boolean equationsPossible(String[] equations) {
        UF uf = new UF(26);
        for (String equation : equations) {
            if (equation.charAt(1) == '=') {
                uf.union(equation.charAt(0) - 'a', equation.charAt(3) - 'a');
            }
        }

        for (String equation : equations) {
            if (equation.charAt(1) == '!') {
                boolean connected = uf.connected(equation.charAt(0) - 'a', equation.charAt(3) - 'a');
                if (connected) {
                    return false;
                }
            }
        }
        return true;
    }

    public static void main(String[] args) {
//        [990].等式方程的可满足性
//        System.out.println(equationsPossible(new String[]{"a==b", "a==d", "a==c"}));
//        System.out.println(equationsPossible(new String[]{"c==c", "b==d", "x!=z"}));

//        [547].省份数量
//        System.out.println(findCircleNum(new int[][]{{1, 1, 0}, {1, 1, 0}, {0, 0, 1}}));
//        System.out.println(findCircleNum(new int[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}));

//        [684].冗余连接
//        System.out.println(Arrays.toString(findRedundantConnection(new int[][]{{1,2}, {2,3}, {3,4}, {1,4}, {1,5}})));
    }


}
