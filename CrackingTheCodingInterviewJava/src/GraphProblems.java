import java.util.ArrayList;
import java.util.HashMap;

public class GraphProblems {
    // ex 4.7

    Project[] findBuildOrder(String[] projects, String[][] dependencies) {
        Graph graph = buildGraph(projects, dependencies);
        return orderProjects(graph.getNodes());
    }

    Graph buildGraph(String[] projects, String[][] dependencies) {
        Graph graph = new Graph();
        for (String project: projects) {
            graph.getOrCreateNode(project);
        }

        for (String[] dependency: dependencies) {
            String first = dependency[0];
            String second = dependency[1];
            graph.addEdge(first, second);
        }

        return graph;
    }

    Project[] orderProjects(ArrayList<Project> projects) {
        Project[] order = new Project[projects.size()];

        // addNonDependent is the core function which actually adds projects to order
        int endOfList = addNonDependent(order, projects, 0);
        int toBeProcessed = 0;
        // "processing" here means removing dependencies from built projects
        while (toBeProcessed < order.length) {
            Project current = order[toBeProcessed];
            if (current == null) {
                return null;
            }
            ArrayList<Project> children = current.getChildren();
            for (Project child: children) {
                child.decrementDependencies();
            }

            endOfList = addNonDependent(order, children, endOfList);
            toBeProcessed++;

        }

        return order;
    }

    int addNonDependent(Project[] order, ArrayList<Project> projects, int offset) {
        for (Project project : projects) {
            if (project.getNumberDepencies() == 0) {
                order[offset] = project;
                offset++;
            }
        }
        return offset;
    }
    public class Graph {
        private ArrayList<Project> nodes = new ArrayList<Project>();
        private HashMap<String, Project> map = new HashMap<String, Project>();

        public Project getOrCreateNode(String name) {
            if (!map.containsKey(name)) {
                Project node = new Project(name);
                nodes.add(node);
                map.put(name, node);
            }

            return map.get(name);
        }

        public void addEdge(String startName, String endName) {
            Project start = getOrCreateNode(startName);
            Project end = getOrCreateNode(endName);
            start.addNeighbor(end);
        }

        public ArrayList<Project> getNodes() {
            return nodes;
        }
    }

    public class Project {
        private ArrayList<Project> children = new ArrayList<Project>();
        private HashMap<String, Project> map = new HashMap<String, Project>();
        private String name;
        private int dependencies = 0;

        public Project(String s) {
            name = s;
        }

        public void addNeighbor(Project node) {
            if (!map.containsKey(node.getName())) {
                children.add(node);
                map.put(node.getName(), node);
                node.incrementDependencies();
            }
        }

        public void incrementDependencies() { dependencies++; }
        public void decrementDependencies() { dependencies--; }

        public String getName() { return name; }
        public ArrayList<Project> getChildren() { return children; }
        public int getNumberDepencies() { return dependencies; }
    }

    // ex 4.8

    BinaryNode commonAncestor(BinaryNode root, BinaryNode p, BinaryNode q) {
        if (!covers(root, p) || !covers(root, q)) {
            return null;
        }
        return ancestorHelper(root, p, q);
    }

    BinaryNode ancestorHelper(BinaryNode root, BinaryNode p, BinaryNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }

        boolean pIsOnLeft = covers(root.left, p);
        boolean qIsOnLeft = covers(root.left, q);
        if (pIsOnLeft != qIsOnLeft) {
            return root;
        }
        BinaryNode childSide = pIsOnLeft ? root.left : root.right;
        return ancestorHelper(childSide, p, q);
    }

    boolean covers(BinaryNode root, BinaryNode p) {
        if (root == null) return false;
        if (root == p) return true;
        return covers(root.left, p) || covers(root.right, p);
    }
}
