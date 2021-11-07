package com.owen.algorithm.v3;

public class UnionFind {

    private int[] size;
    private int[] parent;
    private int count;

    public UnionFind(int x) {
        size = new int[x];
        parent = new int[x];
        count = x;
        for (int i = 0; i < x; i++) {
            parent[i] = i;
            size[i] = 1;
        }
    }

    public boolean connected(int p, int q) {
        int rootP = find(p);
        int rootQ = find(q);
        return rootP == rootQ;
    }

    public int find(int x) {
        while (x != parent[x]) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
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
