from collections import deque

# graph = {
#     'A': ['B', 'C'],
#     'B': ['D', 'E'],
#     'C': ['F'],
#     'D': [],
#     'E': [],
#     'F': []
# }

# def bfs(start_node):
#     visited = set()
#     queue = deque([start_node])
#     print("BFS Traversal:")
    
#     while queue:
#         current = queue.popleft()
#         if current not in visited:
#             print(current, end=' ')
#             visited.add(current)
            
#             for neighbor in graph[current]:
#                 if neighbor not in visited:
#                     queue.append(neighbor)

# bfs('A')


# graph = {
#     'A': ['B', 'C'],
#     'B': ['D', 'E'],
#     'C': ['F'],
#     'D': [],
#     'E': [],
#     'F': []
# }

# def dfs(node, visited=None):
#     if visited is None:
#         visited = set()
    
#     if node not in visited:
#         print(node, end=" ")
#         visited.add(node)
        
#         for neighbor in graph[node]:
#             dfs(neighbor, visited)

# print("\nDFS Traversal: ")
# dfs('A')



graph = {
    'A': ['B', 'C', 'D'],
    'B': ['A', 'E'],
    'C': ['A', 'D', 'F'],
    'D': ['A', 'C', 'E', 'G'],
    'E': ['B', 'D', 'G'],
    'F': ['C', 'G'],
    'G': ['D', 'E', 'F']
}

def bfs(start_node):
    visited = set()
    queue = deque([start_node])
    print("BFS Traversal:")
    
    while queue:
        current = queue.popleft()
        if current not in visited:
            print(current, end=' ')
            visited.add(current)
            
            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append(neighbor)

bfs('A')


