from collections import deque

# Traditional BFS using a queue
def traditional_bfs(G, start):
    visited = {start}
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order

# Traditional DFS using recursion
def traditional_dfs(G, start):
    visited = set()
    order = []
    def dfs(node):
        visited.add(node)
        order.append(node)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                dfs(neighbor)
    dfs(start)
    return order
