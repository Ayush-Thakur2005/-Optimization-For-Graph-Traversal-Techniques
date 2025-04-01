from collections import deque

def bfs(root):
    if not root:
        return
    queue = deque([root])
    traversal = []
    while queue:
        node = queue.popleft()
        traversal.append(node.hash)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return traversal

def dfs(root):
    if not root:
        return []
    return [root.hash] + dfs(root.left) + dfs(root.right)

# Apply BFS and DFS
bfs_result = bfs(merkle_tree.root)
dfs_result = dfs(merkle_tree.root)

print("BFS Traversal:", bfs_result)
print("DFS Traversal:", dfs_result)
