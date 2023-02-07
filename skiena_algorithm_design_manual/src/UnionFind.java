public class UnionFind {
    private int[] parent;
    private int[] size;
    private int n;

    UnionFind(int n) {
        this.parent = new int[n + 1];
        this.size = new int[n + 1];
        this.n = n;
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            size[i] = i;
        }
    }

    public int find(int x) {
        if (parent[x] == x) {
            return x;
        }
        return (find(parent[x]));
    }

    public boolean sameComponent(int x1, int x2) {
        return find(x1) == find(x2);
    }

    public void unionSets(int x1, int x2) {
        int root1 = find(x1);
        int root2 = find(x2);

        if (root1 == root2) {
            return;
        }

        if (size[root1] >= size[root2]) {
            size[root1] += size[root2];
            parent[root2] = root1;
        } else {
            size[root2] += size[root1];
            parent[root1] = root2;
        }
    }
}
