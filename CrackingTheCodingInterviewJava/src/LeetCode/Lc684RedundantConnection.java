package LeetCode;

import java.util.NoSuchElementException;

public class Lc684RedundantConnection {
    public int[] findRedundantConnection(int[][] edges) {
        int[] parents = new int[edges.length + 1];
        for (int i = 0; i < parents.length; i++) {
            parents[i] = i;
        }

        for (int[] edge : edges) {
            int x1 = edge[0];
            int x2 = edge[1];
            if (isConnected(parents, x1, x2)) {
                return edge;
            }
            unionSets(parents, x1, x2);
        }


        throw new NoSuchElementException("Graph has no cycles!");
    }

    // Implements UnionSet without implement UnionSet which in retrospect I think was silly.
    private int find(int[] parents, int x) {
        if (parents[x] == x) {
            return x;
        }

        return find(parents, parents[x]);
    }

    private boolean isConnected(int[] parents, int x1, int x2) {
        return find(parents, x1) == find(parents, x2);
    }

    private void unionSets(int[] parents, int x1, int x2) {
        int r1 = find(parents, x1);
        int r2 = find(parents, x2);

        if (r1 == r2) {
            return;
        }

        parents[r2] = r1;
    }
}
