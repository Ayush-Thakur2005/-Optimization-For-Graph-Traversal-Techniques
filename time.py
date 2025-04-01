import time

def time_function(func, *args, repetitions=100, **kwargs):
    start_time = time.perf_counter()
    for _ in range(repetitions):
        func(*args, **kwargs)
    end_time = time.perf_counter()
    return (end_time - start_time) / repetitions  # Average time per execution

# Choose a starting node (e.g., node 0)
start_node = 0

# Measure execution times for each algorithm
bfs_time = time_function(traditional_bfs, G, start_node)
dfs_time = time_function(traditional_dfs, G, start_node)
opt_bfs_time = time_function(optimized_bfs, G, start_node)

print(f"Traditional BFS average time: {bfs_time:.6f} sec")
print(f"Traditional DFS average time: {dfs_time:.6f} sec")
print(f"Optimized BFS average time:   {opt_bfs_time:.6f} sec")
