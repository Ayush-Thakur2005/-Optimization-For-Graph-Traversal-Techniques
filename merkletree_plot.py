import networkx as nx
import matplotlib.pyplot as plt

def build_graph(G, node, pos, level=0, x=0, width=2.0, y_gap=1.5):
    """ Recursively build the graph for visualization """
    if node is None:
        return

    G.add_node(node.hash, label=node.hash[:6])  # Shorten hash for readability
    pos[node.hash] = (x, -level * y_gap)  # Positioning nodes in levels

    if node.left:
        G.add_edge(node.hash, node.left.hash)
        build_graph(G, node.left, pos, level + 1, x - width / 2, width / 2)
    if node.right:
        G.add_edge(node.hash, node.right.hash)
        build_graph(G, node.right, pos, level + 1, x + width / 2, width / 2)

def draw_merkle_tree(root):
    """ Draws the Merkle Tree using NetworkX """
    G = nx.DiGraph()
    pos = {}
    
    build_graph(G, root, pos)

    plt.figure(figsize=(12, 6))
    labels = {node: G.nodes[node]['label'] for node in G.nodes}
    nx.draw(G, pos, with_labels=True, labels=labels, node_color="lightblue", edge_color="gray", node_size=2000, font_size=8)
    
    plt.title("Merkle Tree Visualization")
    plt.show()

# Draw the Merkle Tree
draw_merkle_tree(merkle_tree.root)
