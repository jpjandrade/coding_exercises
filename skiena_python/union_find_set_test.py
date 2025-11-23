import unittest

from union_find_set import UnionFindSet


class TestUnionFindSet(unittest.TestCase):
    def test_basic_unions_and_find(self):
        uf = UnionFindSet(6)

        # build two components: {0,1,2} and {3,4}
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(3, 4)

        self.assertTrue(uf.same_component(0, 2))
        self.assertFalse(uf.same_component(0, 3))

        # join components
        uf.union(2, 4)
        self.assertTrue(uf.same_component(0, 4))

    def test_component_size_and_idempotent_union(self):
        uf = UnionFindSet(6)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(3, 4)
        uf.union(2, 4)

        self.assertEqual(uf.component_size(0), 5)
        self.assertEqual(uf.component_size(5), 1)

        # idempotent union should not change size
        uf.union(0, 4)
        self.assertEqual(uf.component_size(0), 5)


if __name__ == "__main__":
    unittest.main()
