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
    0: [1, 3, 4],
    1: [2],
    2: [],
    3: [5],
    4: [5],
    5: []
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

bfs(0)



graph = {
    1: [2, 7, 8],
    2: [3, 6],
    3: [4, 5],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [9, 12],
    9: [10, 11],
    10: [],
    11: [],
    12: []
}


def dfs(node, visited=None):
    if visited is None:
        visited = set()
    
    if node not in visited:
        print(node, end=" ")
        visited.add(node)
        
        for neighbor in graph[node]:
            dfs(neighbor, visited)

print("\nDFS Traversal: ")
dfs(1)
