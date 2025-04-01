import networkx as nx
import matplotlib.pyplot as plt

# Create a well-connected synthetic graph with 150 nodes
num_nodes = 150
edge_prob = 0.1  # Higher probability ensures the graph is well-connected
G = nx.erdos_renyi_graph(num_nodes, edge_prob)

print("Graph created with", G.number_of_nodes(), "nodes and", G.number_of_edges(), "edges.")

# Visualize the graph with clearer node separation
plt.figure(figsize=(12, 10))
# Adjust spring_layout: increasing 'k' helps spread nodes out
pos = nx.spring_layout(G, seed=42, k=0.15)

# Draw nodes with a larger size for clarity
nx.draw_networkx_nodes(G, pos, node_size=100, node_color='skyblue', alpha=0.9)
# Draw edges with moderate width and transparency
nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.5)
# Optionally, add labels to nodes for even more clarity (optional)
nx.draw_networkx_labels(G, pos, font_size=8, font_color='darkred')

plt.title("Well-Connected Synthetic Graph (150 nodes, edge probability = 0.1)", fontsize=14)
plt.axis('off')
plt.show()
