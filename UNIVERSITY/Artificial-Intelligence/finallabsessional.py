# #Q1
# #1
# print("Zulqarnain Malik")
# print("Muhammad Ashraf")
# print("19-08-2004")
# #2
# original_string = "Python"
# reversed_string = original_string[::-1]
# print("Original:", original_string)
# print("Reversed:", reversed_string)
# my_string = "Hello World"
# #3
# my_string = "Hello World"
# count = len(my_string)
# print("String:", my_string)
# print("Number of characters:", count)
# #4
# university = "University OF Wah"
# campus = "Quaid Campus"
# semester = "3nd Semester"
# reg_number = "uw-24-cs-bs-043"
# print("University Name:", university)
# print("Campus:", campus)
# print("Class/Semester:", semester)
# print("Registration No:", reg_number)
# #5
# a = 5
# b = 10
# print(f"Before swapping: a = {a}, b = {b}")
# a, b = b, a
# print(f"After swapping:  a = {a}, b = {b}")









# graph = {
#     'A': ['B', 'D', 'C'],
#     'B': ['A', 'E'],
#     'C': ['A', 'F'],
#     'D': ['A', 'C', 'E', 'G'],
#     'E': ['B', 'D', 'G'],
#     'F': ['C', 'G'],
#     'G': ['E', 'D', 'F']
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

# bfs(0)












# import heapq
# def a_star_search(graph, heuristics, start, goal):
#     open_list = []
#     initial_f = 0 + heuristics[start]
#     heapq.heappush(open_list, (initial_f, start, [start], 0))
#     visited_costs = {start: 0} 
#     while open_list:
#         f, current, path, g = heapq.heappop(open_list)
#         if current == goal:
#             return path, g
#         for neighbor, weight in graph.get(current, {}).items():
#             new_g = g + weight   
#             if neighbor not in visited_costs or new_g < visited_costs[neighbor]:
#                 visited_costs[neighbor] = new_g
#                 new_h = heuristics.get(neighbor, 0)
#                 new_f = new_g + new_h
#                 new_path = path + [neighbor]
#                 heapq.heappush(open_list, (new_f, neighbor, new_path, new_g))            
#     return None, float('inf')
# cities_graph = {
#     'Islamabad': {'Peshawar': 174, 'Lahore': 260, 'Quetta': 714, 'Faisalabad': 388},
#     'Peshawar':  {'Sialkot': 260, 'Lahore': 385},
#     'Sialkot':   {'Lahore': 238},
#     'Lahore':    {'Multan': 338, 'Karachi': 437},
#     'Faisalabad':{'Multan': 161},
#     'Quetta':    {'Multan': 388, 'Faisalabad': 161}, 
#     'Multan':    {'Sukkur': 313},
#     'Sukkur':    {'Karachi': 200}￼
# }
# heuristics = {
#     'Islamabad': 125,
#     'Peshawar': 695,
#     'Sialkot': 273,
#     'Lahore': 241,
#     'Faisalabad': 218,
#     'Quetta': 714,
#     'Multan': 357,
#     'Sukkur': 386,
#     'Karachi': 150 
# }
# start_city = 'Islamabad'
# destination_city = 'Karachi'
# optimal_path, total_cost = a_star_search(cities_graph, heuristics, start_city, destination_city)
# print(f"Optimal Route from {start_city} to {destination_city}:")
# print(" -> ".join(optimal_path))
# print(f"Total Travel Cost: {total_cost}")