import matplotlib.pyplot as plt

algorithms = ['Traditional BFS', 'Traditional DFS', 'Optimized BFS']
times = [bfs_time, dfs_time, opt_bfs_time]

plt.figure(figsize=(8, 6))
bars = plt.bar(algorithms, times, color=['blue', 'green', 'red'])
plt.ylabel("Average Execution Time (seconds)")
plt.title("Performance Comparison of Graph Traversal Algorithms")
plt.show()
