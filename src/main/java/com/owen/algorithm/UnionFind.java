package com.owen.algorithm;


import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

    public static class UF2 {
        private int[] parent;
        private double[] weight;

        public UF2(int count) {
            parent = new int[count];
            weight = new double[count];
            for (int i = 0; i < count; i++) {
                parent[i] = i;
                weight[i] = 1.0d;
            }
        }

        public void union(int p, int q, double value) {
            int rootP = find(p);
            int rootQ = find(q);
            if (rootP == rootQ) {
                return;
            }
            parent[rootP] = rootQ;
            weight[rootP] = weight[q] * value / weight[p];
        }

        public int find(int x) {
            if (x != parent[x]) {
                int origin = parent[x];
                //需要递归求解
                parent[x] = find(parent[x]);
                weight[x] *= weight[origin];
            }
            return parent[x];
        }

        public double connected(int p, int q) {
            int rootP = find(p);
            int rootQ = find(q);
            if (rootP == rootQ) {
                return weight[p] / weight[q];
            } else {
                return -1.0d;
            }
        }
    }

    //[399].除法求值
    public static double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        Map<String, Integer> equationIndex = new HashMap<>();

        int index = 0;
        UF2 uf = new UF2(equations.size() * 2);
        for (int i = 0; i < equations.size(); i++) {
            List<String> equation = equations.get(i);
            String firstVar = equation.get(0);
            String secondVar = equation.get(1);
            if (!equationIndex.containsKey(firstVar)) {
                equationIndex.put(firstVar, index++);
            }
            if (!equationIndex.containsKey(secondVar)) {
                equationIndex.put(secondVar, index++);
            }
            uf.union(equationIndex.get(firstVar), equationIndex.get(secondVar), values[i]);
        }

        double[] res = new double[queries.size()];
        for (int i = 0; i < queries.size(); i++) {
            List<String> query = queries.get(i);
            String firstVar = query.get(0);
            String secondVar = query.get(1);
            Integer index1 = equationIndex.get(firstVar);
            Integer index2 = equationIndex.get(secondVar);

            if (index1 == null || index2 == null) {
                res[i] = -1.0d;
            } else {
                res[i] = uf.connected(index1, index2);
            }
        }
        return res;
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

//        double[] res = calcEquation(Arrays.asList(Arrays.asList("a", "b"), Arrays.asList("b", "c")), new double[]{2.0, 3.0}, Arrays.asList(Arrays.asList("a", "c"), Arrays.asList("b", "a"), Arrays.asList("a", "e"), Arrays.asList("a", "a"), Arrays.asList("x", "x")));
        double[] res = calcEquation(Arrays.asList(Arrays.asList("x1", "x2"), Arrays.asList("x2", "x3"), Arrays.asList("x3", "x4"), Arrays.asList("x4", "x5")), new double[]{3.0, 4.0, 5.0, 6.0}, Arrays.asList(Arrays.asList("x1", "x5"), Arrays.asList("x5", "x2"), Arrays.asList("x2", "x4"), Arrays.asList("x2", "x2"), Arrays.asList("x2", "x9"), Arrays.asList("x9", "x9")));
        System.out.println(Arrays.toString(res));
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
