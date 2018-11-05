public class BinaryNode {
    public int data;
    public BinaryNode left;
    public BinaryNode right;

    public BinaryNode(int d) {
        data = d;
    }

    public void inOrderTraversal(BinaryNode node) {
        if (node != null) {
            inOrderTraversal(node.left);
            System.out.println(node.data);
            inOrderTraversal(node.right);
        }
    }
}

