# import heapq  

# graph = {
#     'A': [('B', 1), ('C', 5)],
#     'B': [('D', 7)],
#     'C': [('E', 8)],
#     'D': [('E', 10)],
#     'E': []  
# }

# heuristics = {
#     'A': 8,
#     'B': 7,
#     'C': 5,
#     'D': 3,
#     'E': 0
# }

# def a_star_algorithm(graph, heuristics, start, goal):
#     open_list = [] 
    
#     heapq.heappush(open_list, (heuristics[start], 0, start, [start]))

#     while open_list:
#         f, g, current, path = heapq.heappop(open_list) 

#         if current == goal:
#             return path, g

#         for neighbor, cost in graph[current]:
#             new_g = g + cost                       
#             new_f = new_g + heuristics[neighbor]    
            
#             heapq.heappush(open_list, (new_f, new_g, neighbor, path + [neighbor]))

#     return None, float('inf') 

# path, cost = a_star_algorithm(graph, heuristics, 'A', 'E')

# print("Optimal Path:", path)
# print("Total Cost:", cost)




# Lab Task:
# You are working with a warehouse robot that must plan the shortest path from the entry
# gate to the delivery zone. Design your own city/warehouse map using a graph, assign
# heuristic values, and use A* to calculate the most efficient route. Your graph should have at
# least 6 nodes and contain multiple possible paths. Display the final path and total cost.

import heapq

warehouse_map = {
    'A': [('B', 3), ('C', 5)],
    'B': [('D', 4), ('E', 6)],
    'C': [('E', 4)],
    'D': [('F', 5)],
    'E': [('F', 2)],
    'F': []
}

heuristics = {
    'A': 10,
    'B': 7,
    'C': 6,
    'D': 4,
    'E': 2,
    'F': 0
}

def warehouse_robot_path(graph, heuristics, start, goal):

    open_list = []
    heapq.heappush(open_list, (heuristics[start], 0, start, [start]))
    
    visited = {} 

    while open_list:
        f, g, current, path = heapq.heappop(open_list)

        if current == goal:
            return path, g

        if current in visited and visited[current] <= g:
            continue
        visited[current] = g

        for neighbor, cost in graph[current]:
            new_g = g + cost
            new_f = new_g + heuristics[neighbor]
            heapq.heappush(open_list, (new_f, new_g, neighbor, path + [neighbor]))

    return None, float('inf')

path, total_cost = warehouse_robot_path(warehouse_map, heuristics, 'A', 'F')

print(f"Optimal Path:  {' -> '.join(path)}")
print(f"Total Distance: {total_cost} meters")