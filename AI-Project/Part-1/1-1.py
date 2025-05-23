#Implement and compare the running time of DFS,BFS, UCS and A* search algorithms on an example of your design.


import time
import heapq

# Define the graph as adjacency list with costs
graph = {
    'A': [('B', 1), ('D', 5)],
    'B': [('E', 2)],
    'D': [('G', 2)],
    'E': [('G', 4)],
    'G': []
}

# Heuristic function for A* (estimated cost from each node to goal 'G')
heuristic = {
    'A': 6,
    'B': 5,
    'D': 2,
    'E': 4,
    'G': 0
}

def dfs(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    while stack:
        (vertex, path) = stack.pop()
        if vertex == goal:
            return path
        if vertex not in visited:
            visited.add(vertex)
            for (neighbor, _) in reversed(graph[vertex]):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    return None

def bfs(graph, start, goal):
    queue = [(start, [start])]
    visited = set()
    while queue:
        (vertex, path) = queue.pop(0)
        if vertex == goal:
            return path
        if vertex not in visited:
            visited.add(vertex)
            for (neighbor, _) in graph[vertex]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    return None

def ucs(graph, start, goal):
    queue = []
    heapq.heappush(queue, (0, start, [start]))
    visited = set()
    while queue:
        (cost, vertex, path) = heapq.heappop(queue)
        if vertex == goal:
            return (path, cost)
        if vertex not in visited:
            visited.add(vertex)
            for (neighbor, weight) in graph[vertex]:
                if neighbor not in visited:
                    heapq.heappush(queue, (cost + weight, neighbor, path + [neighbor]))
    return None

def a_star(graph, start, goal, heuristic):
    queue = []
    heapq.heappush(queue, (heuristic[start], 0, start, [start]))
    visited = set()
    while queue:
        (est_total_cost, cost_so_far, vertex, path) = heapq.heappop(queue)
        if vertex == goal:
            return (path, cost_so_far)
        if vertex not in visited:
            visited.add(vertex)
            for (neighbor, weight) in graph[vertex]:
                if neighbor not in visited:
                    new_cost = cost_so_far + weight
                    est = new_cost + heuristic[neighbor]
                    heapq.heappush(queue, (est, new_cost, neighbor, path + [neighbor]))
    return None

if __name__ == "__main__":
    start_node = 'A'
    goal_node = 'G'

    start_time = time.time()
    dfs_result = dfs(graph, start_node, goal_node)
    dfs_time = time.time() - start_time
    print("DFS result:", dfs_result)
    print(f"DFS runtime: {dfs_time:.6f} seconds\n")

    start_time = time.time()
    bfs_result = bfs(graph, start_node, goal_node)
    bfs_time = time.time() - start_time
    print("BFS result:", bfs_result)
    print(f"BFS runtime: {bfs_time:.6f} seconds\n")

    start_time = time.time()
    ucs_result = ucs(graph, start_node, goal_node)
    ucs_time = time.time() - start_time
    print("UCS result:", ucs_result)
    print(f"UCS runtime: {ucs_time:.6f} seconds\n")

    start_time = time.time()
    astar_result = a_star(graph, start_node, goal_node, heuristic)
    astar_time = time.time() - start_time
    print("A* result:", astar_result)
    print(f"A* runtime: {astar_time:.6f} seconds\n")
