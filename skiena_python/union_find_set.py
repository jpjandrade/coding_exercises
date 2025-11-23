class UnionFindSet:
    """Disjoint-set (Union-Find) with path compression and union-by-size.

    This implementation keeps a `parent` array where `parent[i] == i` indicates
    a root. `size[root]` stores the number of elements in the component whose
    root is `root`.

    Both optimizations are used:
    - Path compression (in `find`): flattens trees by making nodes point
      directly to their root during finds, which reduces future find time.
    - Union-by-size (in `union`): always attach the smaller tree under the
      larger tree, keeping tree height small.
    """

    def __init__(self, n):
        """Create a UnionFind for `n` elements (0..n-1)."""
        self.n = n
        self.parent = [i for i in range(n)]
        self.size = [1 for _ in range(n)]

    def find(self, x):
        """Return the root of `x`, applying full path compression.

        If `x` is not a root, recursively find the root and set
        `parent[x]` to that root. This flattens the path from `x` to the root,
        giving near-constant amortized time per operation (inverse-Ackermann).
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union the sets containing `x` and `y` using union-by-size.

        Finds the roots of `x` and `y`. If they differ, attach the smaller
        root under the larger root and update the `size` of the new root.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # Ensure root_x is the root of the larger component.
        if self.size[root_x] < self.size[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]

    def same_component(self, x, y):
        return self.find(x) == self.find(y)

    def component_size(self, x):
        """Return the size of the component that contains `x`."""
        return self.size[self.find(x)]
