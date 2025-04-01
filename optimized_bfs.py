def optimized_bfs(G, start, threshold=0.3):
    n = G.number_of_nodes()
    visited = {start}
    frontier = {start}
    order = [start]
    while frontier:
        if len(frontier) < threshold * n:
            # Top-Down Approach: Expand from nodes in the current frontier.
            new_frontier = set()
            for node in frontier:
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        new_frontier.add(neighbor)
                        visited.add(neighbor)
                        order.append(neighbor)
            frontier = new_frontier
        else:
            # Bottom-Up Approach: Iterate over all unvisited nodes and check if any neighbor is in the frontier.
            new_frontier = set()
            for node in G.nodes():
                if node not in visited:
                    for neighbor in G.neighbors(node):
                        if neighbor in frontier:
                            new_frontier.add(node)
                            visited.add(node)
                            order.append(node)
                            break
            frontier = new_frontier
    return order
