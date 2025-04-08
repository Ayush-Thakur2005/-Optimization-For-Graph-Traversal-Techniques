import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import time
from collections import deque
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Step 1: Generate a random graph with 150 nodes and edge probability 0.1
n = 150  # number of nodes
p = 0.1  # edge probability
G = nx.gnp_random_graph(n, p)

# Step 2: BFS Implementation and Performance Measurement
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order

# Measure BFS performance
start_node = random.choice(list(G.nodes()))
start_time = time.time()
bfs_order = bfs(G, start_node)
bfs_time = time.time() - start_time
bfs_nodes_visited = len(bfs_order)
print(f"BFS Results:")
print(f"  Execution Time: {bfs_time:.6f} seconds")
print(f"  Nodes Visited: {bfs_nodes_visited}/{n}")

# Step 3: GNN Implementation and Performance Measurement

# Assign synthetic labels (1 if degree > median, 0 otherwise)
degrees = [d for _, d in G.degree()]
median_degree = np.median(degrees)
labels = [1 if G.degree(node) > median_degree else 0 for node in G.nodes()]

# Create node features (degree as a feature)
features = torch.tensor([[G.degree(node)] for node in G.nodes()], dtype=torch.float)

# Convert to PyTorch Geometric format
edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
data = Data(x=features, edge_index=edge_index, y=torch.tensor(labels, dtype=torch.long))

# Split into train/test masks (80% train, 20% test)
train_mask = torch.zeros(n, dtype=torch.bool)
test_mask = torch.zeros(n, dtype=torch.bool)
indices = list(range(n))
random.shuffle(indices)
train_size = int(0.8 * n)
train_mask[indices[:train_size]] = True
test_mask[indices[train_size:]] = True
data.train_mask = train_mask
data.test_mask = test_mask

# Define GNN model (Graph Convolutional Network)
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, 16)  # 1 input feature -> 16 hidden
        self.conv2 = GCNConv(16, 2)  # 16 hidden -> 2 classes

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Train and evaluate GNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Measure training time
start_time = time.time()
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
gnn_train_time = time.time() - start_time

# Measure inference time and accuracy
model.eval()
start_time = time.time()
with torch.no_grad():
    out = model(data)
    _, pred = out.max(dim=1)
gnn_inference_time = time.time() - start_time
gnn_accuracy = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

print(f"\nGNN Results:")
print(f"  Training Time (200 epochs): {gnn_train_time:.6f} seconds")
print(f"  Inference Time: {gnn_inference_time:.6f} seconds")
print(f"  Classification Accuracy: {gnn_accuracy:.4f}")

# Step 4: Visualize Performance Comparison
plt.figure(figsize=(10, 5))

# Bar plot for execution times
plt.subplot(1, 2, 1)
methods = ['BFS', 'GNN Train', 'GNN Infer']
times = [bfs_time, gnn_train_time, gnn_inference_time]
bars = plt.bar(methods, times, color=['blue', 'green', 'orange'])
plt.ylabel('Time (seconds)')
plt.title('Execution Time Comparison')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

# Bar plot for task-specific metric (nodes visited vs accuracy)
plt.subplot(1, 2, 2)
metrics = ['BFS Nodes Visited', 'GNN Accuracy']
values = [bfs_nodes_visited / n, gnn_accuracy]  # Normalize nodes visited to [0,1]
bars = plt.bar(metrics, values, color=['blue', 'green'])
plt.ylabel('Normalized Metric (0-1)')
plt.title('Task Performance')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
