# # Depth-Limited Search (DLS)

# def dls(graph, current, goal, depth, path, visited):
#     if depth == 0:
#         if current == goal:
#             return path + [current]
#         return None
#     visited.add(current)
#     if current == goal:
#         return path + [current]
#     for neighbor in graph.get(current, []):
#         if neighbor not in visited:
#             result = dls(graph, neighbor, goal, depth - 1, path + [current], visited.copy())
#             if result:
#                 return result
#     return None

# # Iterative Deepening Depth-First Search (IDDFS)
# def iddfs(graph, start, goal, max_depth=10):
#     for depth in range(max_depth):
#         visited = set()
#         path = dls(graph, start, goal, depth, [], visited)
#         if path:
#             return path
#     return None

# graph = {
#     'A': ['B', 'C'],
#     'B': ['D', 'E'],
#     'C': ['F'],
#     'D': [],
#     'E': ['F'],
#     'F': []
# }
# start_node = 'A'
# goal_node = 'F'

# start = start_node
# goal = goal_node

# result = iddfs(graph, start_node, goal_node)
# print("Path found:", result)

#-----------------------------------------------------------------------------------------------------------------------

# Question 1: Pathfinding in a Simple Graph 


# Scenario: You are given an unweighted, undirected graph represented as an adjacency list. 


# Implement IDDFS to find a path from a start node to a goal node. 


# Input: 

# Python

# graph = {
# 'A': ['B', 'C'],
# 'B': ['D', 'E'],
# 'C': ['F'],
# 'D': [],
# 'E': ['F'],
# 'F': []
# }
# start = 'A'
# goal = 'F'
# ``` [cite: 98-107]

# Task: Implement a function iddfs(graph, start, goal) that returns a path (as a list of nodes) from start to goal using IDDFS. 


# Question 2: Word Ladder Puzzle (One-Letter Change) 

# Scenario: You're given a dictionary of words. Your goal is to transform a start word into a goal word by changing one letter at a time, and each intermediate word must also be in the dictionary. 


# Input: 

# Python

# word_list = ['hit', 'hot', 'dot', 'dog', 'cog', 'log', 'lot']
# start = 'hit'
# goal = 'cog'
# ``` [cite: 112-114]

# Task: Write a function iddfs_word_ladder(start, goal, word_list) that finds a transformation path using IDDFS. 


# Question 3: 8-Puzzle Solver (Simplified) 


# Scenario: Given a simplified 8-puzzle configuration (3x3 grid), where 0 represents the blank space, use IDDFS to find the sequence of moves to reach the goal configuration. 


# Input: 

# Python

# start_state = [[1, 2, 3],
#                [4, 0, 6],
#                [7, 5, 8]]
# goal_state = [[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 0]]
# ``` [cite: 120-125]

# Task: Implement iddfs_puzzle(start_state, goal_state) to return the list of states or moves to solve the puzzle using IDDFS. 


# Question 4: Maze Navigation 


# Scenario: Given a 2D grid maze where 0 is a wall and 1 is a valid path, use IDDFS to find a path from a starting position to a goal position. 


# Input: 

# Python

# maze = [
# [1, 0, 1, 1],
# [1, 1, 0, 1],
# [0, 1, 1, 0],
# [1, 0, 1, 1]
# ]
# start = (0, 0)
# goal = (3, 3)
# ``` [cite: 130-137]

# Task: Write iddfs_maze(maze, start, goal) to return a list of positions that lead from start to goal.

#-----------------------------------------------------------------------------------------------------------------------

#Question 1: Pathfinding in a Simple Graph

# def dls(graph, current, goal, depth, path, visited):
#     if depth == 0:
#         if current == goal:
#             return path + [current]
#         return None
#     visited.add(current)
#     if current == goal:
#         return path + [current]
#     for neighbor in graph.get(current, []):
#         if neighbor not in visited:
#             result = dls(graph, neighbor, goal, depth - 1, path + [current], visited.copy())
#             if result:
#                 return result
#     return None

# def iddfs(graph, start, goal, max_depth=10):
#     for depth in range(max_depth):
#         visited = set()
#         path = dls(graph, start, goal, depth, [], visited)
#         if path:
#             return path
#     return None

# graph = {
#     'A': ['B', 'C'],
#     'B': ['D', 'E'],
#     'C': ['F'],
#     'D': [],
#     'E': ['F'],
#     'F': []
# }

# start_node = 'A'
# goal_node = 'F'

# start = start_node
# goal = goal_node

# result = iddfs(graph, start_node, goal_node)
# print("Path found:", result)

#-----------------------------------------------------------------------------------------------------------------------

#Question 2: Word Ladder Puzzle (One-Letter Change)

# def one_letter_diff(word1, word2):
#     diff_count = sum(c1 != c2 for c1, c2 in zip(word1, word2))
#     return diff_count == 1

# def dls_word_ladder(current, goal, word_list, depth, path, visited):
#     if depth == 0:
#         if current == goal:
#             return path + [current]
#         return None
#     visited.add(current)
#     if current == goal:
#         return path + [current]
#     for word in word_list:
#         if word not in visited and one_letter_diff(current, word):
#             result = dls_word_ladder(word, goal, word_list, depth - 1, path + [current], visited.copy())
#             if result:
#                 return result
#     return None

# def iddfs_word_ladder(start, goal, word_list, max_depth=10):
#     for depth in range(max_depth):
#         visited = set()
#         path = dls_word_ladder(start, goal, word_list, depth, [], visited)
#         if path:
#             return path
#     return None

# word_list = ['hit', 'hot', 'dot', 'dog', 'cog', 'log', 'lot']

# start_word = 'hit'
# goal_word = 'cog'
 
# result = iddfs_word_ladder(start_word, goal_word, word_list)
# print("Path found:", result)

#-----------------------------------------------------------------------------------------------------------------------

#Question 3: 8-Puzzle Solver (Simplified)

# def get_neighbors(state):
#     neighbors = []
#     zero_pos = [(i, row.index(0)) for i, row in enumerate(state) if 0 in row][0]
#     x, y = zero_pos
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#     for dx, dy in directions:
#         new_x, new_y = x + dx, y + dy
#         if 0 <= new_x < 3 and 0 <= new_y < 3:
#             new_state = [list(row) for row in state]
#             new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
#             neighbors.append(new_state)
#     return neighbors

# def dls_puzzle(current, goal, depth, path, visited):
#     if depth == 0:
#         if current == goal:
#             return path + [current]
#         return None
#     visited.add(tuple(map(tuple, current)))
#     if current == goal:
#         return path + [current]
#     for neighbor in get_neighbors(current):
#         if tuple(map(tuple, neighbor)) not in visited:
#             result = dls_puzzle(neighbor, goal, depth - 1, path + [current], visited.copy())
#             if result:
#                 return result
#     return None

# def iddfs_puzzle(start, goal, max_depth=20):
#     for depth in range(max_depth):
#         visited = set()
#         path = dls_puzzle(start, goal, depth, [], visited)
#         if path:
#             return path
#     return None

# start_state = [[1, 2, 3],
#                [4, 0, 6],
#                [7, 5, 8]]
# goal_state = [[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 0]]

# result = iddfs_puzzle(start_state, goal_state)
# print("Path found:")
# for state in result:
#     for row in state:
#         print(row)
#     print()

#-----------------------------------------------------------------------------------------------------------------------

#Question 4: Maze Navigation

# def get_maze_neighbors(maze, position):
#     neighbors = []
#     x, y = position
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#     for dx, dy in directions:
#         new_x, new_y = x + dx, y + dy
#         if 0 <= new_x < len(maze) and 0 <= new_y < len(maze[0]) and maze[new_x][new_y] == 1:
#             neighbors.append((new_x, new_y))
#     return neighbors

# def dls_maze(current, goal, depth, path, visited):
#     if depth == 0:
#         if current == goal:
#             return path + [current]
#         return None
#     visited.add(current)
#     if current == goal:
#         return path + [current]
#     for neighbor in get_maze_neighbors(maze, current):
#         if neighbor not in visited:
#             result = dls_maze(neighbor, goal, depth - 1, path + [current], visited.copy())
#             if result:
#                 return result
#     return None

# def iddfs_maze(maze, start, goal, max_depth=20):
#     for depth in range(max_depth):
#         visited = set()
#         path = dls_maze(start, goal, depth, [], visited)
#         if path:
#             return path
#     return None

# maze = [
#     [1, 0, 1, 1],
#     [1, 1, 0, 1],
#     [0, 1, 1, 0],
#     [1, 0, 1, 1]
# ]

# start_pos = (0, 0)
# goal_pos = (3, 3)

# result = iddfs_maze(maze, start_pos, goal_pos)
# print("Path found:", result)