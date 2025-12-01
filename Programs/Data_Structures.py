# # Create a list
# numbers = [10, 20, 30, 40]

# # Access elements
# print(numbers[0])   # 10
# print(numbers[-1])  # 40 (last)

# # Modify elements
# numbers[1] = 25

# # Add elements
# numbers.append(50)
# numbers.insert(2, 99)

# # Remove elements
# numbers.remove(30)
# last = numbers.pop()   # removes last item

# # Loop through list
# for n in numbers:
#     print(n)

# # Length of list
# print("Total items:", len(numbers))

#-------------------------------------------------------------------------------

# Make a list of your 5 favorite movies
# Then print each one with a numbered list

# movies = ["Interstellar", "Inception", "Dune", "Joker", "Avengers"]

# for i in range(len(movies)):
#     print(f"{i+1}. {movies[i]}")

#-------------------------------------------------------------------------------

# games = ["Chess", "Monopoly", "Scrabble", "Poker"]

# print(games[0])  # Chess
# print(games[-1])  # Poker (last)

# games[1] = "Risk"  # Modify second item

# games.append("Clue")  # Add new game at the end
# games.insert(3, "Uno")  # Insert Uno at index 3

# games.remove("Poker")  # Remove Poker

# last_game = games.pop()  # Remove last game (Clue)

# for g in games:
#     print(g)

# print("Total games:", len(games))

# for i in range(len(games)):
#     print(f"{i+1}. {games[i]}")

#-------------------------------------------------------------------------------

# # Creating a tuple
# info = ("Saadi", 19, "Python Developer")

# # Accessing elements
# print(info[0])   # Saadi
# print(info[-1])  # Python Developer

# # Length
# print("Length:", len(info))

#-------------------------------------------------------------------------------

# def get_coordinates():
#     return (10, 20)

# x, y = get_coordinates()
# print("X:", x)
# print("Y:", y)

#-------------------------------------------------------------------------------

# # Your birth info
# birth_info = ("Saadi", "2006", "Islamabad", "Pakistan")

# # Print each item with a label
# print("Name:", birth_info[0])
# print("Year:", birth_info[1])
# print("City:", birth_info[2])
# print("Country:", birth_info[3])

# # Check length
# print("Items in tuple:", len(birth_info))

#-------------------------------------------------------------------------------

# single = (5,)
# print("Single item tuple:", len(single))

#-------------------------------------------------------------------------------

# # Creating a set
# my_set = {1, 2, 3, 4, 4, 5}

# print(my_set)  # Output: {1, 2, 3, 4, 5} — duplicate 4 removed

# # Add items
# my_set.add(6)

# # Remove item
# my_set.remove(2)

# # Check membership
# print(3 in my_set)   # True
# print(9 in my_set)   # False

# # Loop through a set
# for item in my_set:
#     print(item)

#-------------------------------------------------------------------------------

# # Remove duplicates from a list
# nums = [1, 2, 2, 3, 4, 4, 5]
# unique_nums = set(nums)
# print("Unique:", unique_nums)

#-------------------------------------------------------------------------------

# # Make a set of your hobbies (add duplicates)
# hobbies = {"reading", "coding", "gaming", "gaming", "reading", "reading", "coding", "scrolling", "puzzle solving"}

# # Print all unique hobbies
# print("Your unique hobbies:")
# for h in hobbies:
#     print("-", h)

# # Add a new hobby
# hobbies.add("traveling")

# # Remove one
# hobbies.remove("coding")

# print("Updated hobbies:", hobbies)

# print("Your unique hobbies:")
# for h in hobbies:
#     print("-", h)

#-------------------------------------------------------------------------------

# # Create a dictionary
# student = {
#     "name": "Saadi",
#     "age": 19,
#     "grade": "A"
# }

# # Access values
# print(student["name"])   # Saadi
# print(student.get("age"))  # 19

# # Add/update values
# student["city"] = "Lahore"
# student["age"] = 20

# # Remove a key
# student.pop("grade")

# # Loop through keys and values
# for key, value in student.items():
#     print(f"{key}: {value}")

#-------------------------------------------------------------------------------

# # Make a dictionary of yourself
# profile = {
#     "name": "Saadi",
#     "age": 19,
#     "country": "Pakistan",
#     "skills": ["Python", "Java", "SQL"]
# }

# # Print all details
# for key, value in profile.items():
#     print(f"{key.title()}: {value}")

# # Add a new key
# profile["email"] = "saadi@example.com"

# # Remove a key
# profile.pop("country")

# print("\nUpdated Profile:")
# for k, v in profile.items():
#     print(f"{k}: {v}")

#-------------------------------------------------------------------------------

# # Student Database using dict, list, set, and tuple

# students = {}

# def add_student():
#     roll = input("Enter Roll Number: ")
#     name = input("Enter Name: ")
#     age = int(input("Enter Age: "))

#     subjects = input("Enter subjects (comma-separated): ").split(",")
#     skills = set(input("Enter skills (space-separated): ").split())

#     # Save as dictionary entry
#     students[roll] = {
#         "name": name,
#         "age": age,
#         "subjects": [sub.strip() for sub in subjects],
#         "skills": skills
#     }

#     print(f"✅ Student {name} added.\n")

# def show_students():
#     if not students:
#         print("❌ No students yet.")
#         return

#     print("\n📋 All Students:")
#     for roll, info in students.items():
#         print(f"\n🔹 Roll No: {roll}")
#         print(f"   Name: {info['name']}")
#         print(f"   Age: {info['age']}")
#         print(f"   Subjects: {', '.join(info['subjects'])}")
#         print(f"   Skills: {', '.join(info['skills'])}")

# def remove_student():
#     roll = input("Enter Roll Number to remove: ")
#     if roll in students:
#         del students[roll]
#         print(f"✅ Student {roll} removed.")
#     else:
#         print("❌ No such student found.")

# # --- Menu ---
# while True:
#     print("\n📚 Student Database Menu:")
#     print("1. Add Student")
#     print("2. View Students")
#     print("3. Remove Student")
#     print("4. Exit")

#     choice = input("Choose an option (1-4): ")

#     if choice == '1':
#         add_student()
#     elif choice == '2':
#         show_students()
#     elif choice == '3':
#         remove_student()
#     elif choice == '4':
#         print("Exiting... 👋")
#         break
#     else:
#         print("❌ Invalid choice. Try again.")

#-------------------------------------------------------------------------------

# # 🔍 Search Algorithms 

# def linear_search(arr, target):
#     for i in range(len(arr)):
#         if arr[i] == target:
#             return i  # return index if found
#     return -1  # not found

# # Example usage
# names = ["Ali", "Sara", "Saadi", "Ayesha"]
# target = "Saadi"

# result = linear_search(names, target)

# if result != -1:
#     print(f"✅ '{target}' found at index {result}")
# else:
#     print(f"❌ '{target}' not found.")

#-------------------------------------------------------------------------------

# def find_number(numbers, x):
#     for i in range(len(numbers)):
#         if numbers[i] == x:
#             print(f"{x} found at index {i}")
#             return
#     print(f"{x} not found in the list")

# nums = [12, 5, 9, 23, 88, 42]
# find_number(nums, 23)
# find_number(nums, 100)

#-------------------------------------------------------------------------------

# def binary_search(arr, target):
#     low = 0
#     high = len(arr) - 1

#     while low <= high:
#         mid = (low + high) // 2

#         if arr[mid] == target:
#             return mid
#         elif arr[mid] < target:
#             low = mid + 1
#         else:
#             high = mid - 1

#     return -1

# # Example usage
# nums = [5, 10, 15, 20, 25, 30]
# target = 25

# result = binary_search(nums, target)

# if result != -1:
#     print(f"✅ {target} found at index {result}")
# else:
#     print(f"❌ {target} not found.")

#-------------------------------------------------------------------------------

# def binary_search(arr, target):
#     low = 0
#     high = len(arr) - 1

#     while low <= high:
#         mid = (low + high) // 2

#         if arr[mid] == target:
#             return mid
#         elif arr[mid] < target:
#             low = mid + 1
#         else:
#             high = mid - 1

#     return -1

# def find_roll(rolls, target):
#     rolls.sort()  # Make sure it's sorted
#     result = binary_search(rolls, target)

#     if result != -1:
#         print(f"Roll number {target} found at index {result}")
#     else:
#         print("Roll number not found.")

# # Example run
# roll_numbers = [104, 101, 105, 102, 100, 103]
# find_roll(roll_numbers, 102)
# find_roll(roll_numbers, 999)

#-------------------------------------------------------------------------------

# students = []

# # Add student
# def add_student():
#     roll = int(input("Enter Roll Number: "))
#     name = input("Enter Student Name: ")
#     students.append({"roll": roll, "name": name})
#     print("✅ Student added.\n")

# # Show all
# def show_all():
#     if not students:
#         print("No students found.")
#         return
#     print("\n📋 Student Records:")
#     for s in students:
#         print(f"Roll No: {s['roll']} | Name: {s['name']}")
#     print()

# # Linear Search
# def linear_search(roll):
#     for i, s in enumerate(students):
#         if s['roll'] == roll:
#             return i
#     return -1

# # Binary Search
# def binary_search(sorted_list, target):
#     low = 0
#     high = len(sorted_list) - 1
#     while low <= high:
#         mid = (low + high) // 2
#         if sorted_list[mid]['roll'] == target:
#             return mid
#         elif sorted_list[mid]['roll'] < target:
#             low = mid + 1
#         else:
#             high = mid - 1
#     return -1

# # Binary search wrapper
# def binary_search_student(roll):
#     sorted_list = sorted(students, key=lambda x: x['roll'])
#     index = binary_search(sorted_list, roll)
#     if index != -1:
#         print(f"✅ Found: Roll {sorted_list[index]['roll']}, Name: {sorted_list[index]['name']}")
#     else:
#         print("❌ Student not found.")

# # Menu
# while True:
#     print("\n===== Student Finder Menu =====")
#     print("1. Add Student")
#     print("2. View All Students")
#     print("3. Search by Roll (Linear Search)")
#     print("4. Search by Roll (Binary Search)")
#     print("5. Exit")

#     choice = input("Choose an option: ")

#     if choice == '1':
#         add_student()
#     elif choice == '2':
#         show_all()
#     elif choice == '3':
#         roll = int(input("Enter roll number to search: "))
#         index = linear_search(roll)
#         if index != -1:
#             s = students[index]
#             print(f"✅ Found: Roll {s['roll']}, Name: {s['name']}")
#         else:
#             print("❌ Student not found.")
#     elif choice == '4':
#         roll = int(input("Enter roll number to search: "))
#         binary_search_student(roll)
#     elif choice == '5':
#         print("👋 Exiting...")
#         break
#     else:
#         print("❌ Invalid choice.")

#-------------------------------------------------------------------------------

# def bubble_sort(arr):
#     n = len(arr)

#     for i in range(n):  # total passes
#         for j in range(0, n - i - 1):  # adjacent pairs
#             if arr[j] > arr[j + 1]:  # swap if wrong order
#                 arr[j], arr[j + 1] = arr[j + 1], arr[j]

# # Example usage
# nums = [64, 34, 25, 12, 22, 11, 90]
# bubble_sort(nums)
# print("Sorted list:", nums)

#-------------------------------------------------------------------------------

# def insertion_sort(arr):
#     for i in range(1, len(arr)):
#         key = arr[i]           # current item
#         j = i - 1

#         # Shift elements greater than key to the right
#         while j >= 0 and arr[j] > key:
#             arr[j + 1] = arr[j]
#             j -= 1

#         arr[j + 1] = key       # insert key in correct spot

# # numbers = [12, 11, 13, 5, 6]
# # insertion_sort(numbers)
# # print("Sorted list:", numbers)

# grades = [72, 88, 69, 95, 55, 78]
# insertion_sort(grades)
# print("Sorted grades:", grades)

#-------------------------------------------------------------------------------

# def selection_sort(arr):
#     n = len(arr)

#     for i in range(n):
#         min_idx = i  # assume current is smallest

#         for j in range(i+1, n):
#             if arr[j] < arr[min_idx]:
#                 min_idx = j  # update if smaller found

#         # Swap
#         arr[i], arr[min_idx] = arr[min_idx], arr[i]

# # # Example
# # nums = [64, 25, 12, 22, 11]
# # selection_sort(nums)
# # print("Sorted list:", nums)

# heights = [170, 150, 180, 165, 160]
# selection_sort(heights)
# print("Sorted heights:", heights)

#-------------------------------------------------------------------------------

# students = []

# # Add student
# def add_student():
#     name = input("Enter Student Name: ")
#     marks = int(input("Enter Marks: "))
#     students.append({"name": name, "marks": marks})
#     print("✅ Student added.\n")

# # Show all
# def view_students():
#     if not students:
#         print("❌ No students found.")
#         return
#     print("\n📋 Student List:")
#     for s in students:
#         print(f"{s['name']} - {s['marks']}")

# # Bubble Sort by marks
# def bubble_sort():
#     arr = students.copy()
#     n = len(arr)
#     for i in range(n):
#         for j in range(n - i - 1):
#             if arr[j]['marks'] > arr[j + 1]['marks']:
#                 arr[j], arr[j + 1] = arr[j + 1], arr[j]
#     print("\n📊 Sorted by Bubble Sort:")
#     for s in arr:
#         print(f"{s['name']} - {s['marks']}")

# # Insertion Sort by marks
# def insertion_sort():
#     arr = students.copy()
#     for i in range(1, len(arr)):
#         key = arr[i]
#         j = i - 1
#         while j >= 0 and arr[j]['marks'] > key['marks']:
#             arr[j + 1] = arr[j]
#             j -= 1
#         arr[j + 1] = key
#     print("\n📊 Sorted by Insertion Sort:")
#     for s in arr:
#         print(f"{s['name']} - {s['marks']}")

# # Selection Sort by marks
# def selection_sort():
#     arr = students.copy()
#     n = len(arr)
#     for i in range(n):
#         min_idx = i
#         for j in range(i+1, n):
#             if arr[j]['marks'] < arr[min_idx]['marks']:
#                 min_idx = j
#         arr[i], arr[min_idx] = arr[min_idx], arr[i]
#     print("\n📊 Sorted by Selection Sort:")
#     for s in arr:
#         print(f"{s['name']} - {s['marks']}")

# # Main menu
# while True:
#     print("\n===== Topper Board Menu =====")
#     print("1. Add Student")
#     print("2. View All")
#     print("3. Bubble Sort by Marks")
#     print("4. Insertion Sort by Marks")
#     print("5. Selection Sort by Marks")
#     print("6. Exit")

#     choice = input("Enter choice: ")

#     if choice == '1':
#         add_student()
#     elif choice == '2':
#         view_students()
#     elif choice == '3':
#         bubble_sort()
#     elif choice == '4':
#         insertion_sort()
#     elif choice == '5':
#         selection_sort()
#     elif choice == '6':
#         print("👋 Exiting...")
#         break
#     else:
#         print("❌ Invalid option.")

#-------------------------------------------------------------------------------

# class Stack:
#     def __init__(self):
#         self.items = []

#     def push(self, item):
#         self.items.append(item)

#     def pop(self):
#         if self.is_empty():
#             return "❌ Stack is empty."
#         return self.items.pop()

#     def peek(self):
#         if self.is_empty():
#             return "❌ Stack is empty."
#         return self.items[-1]

#     def is_empty(self):
#         return len(self.items) == 0

#     def display(self):
#         print("📦 Stack (top to bottom):")
#         for item in reversed(self.items):
#             print(item)

# s = Stack()

# s.push("Book")
# s.push("Notebook")
# s.push("Laptop")
# s.push("Pen")
# s.push("Phone")
# s.push("Charger")
# s.push("Headphones")
# s.push("Mouse")
# s.display()

# print("Top item:", s.peek())
# print("Popped:", s.pop())
# s.display()

# print("Top item:", s.peek())
# print("Popped:", s.pop())
# s.display()

#-------------------------------------------------------------------------------

# class Queue:
#     def __init__(self):
#         self.items = []

#     def enqueue(self, item):
#         self.items.append(item)

#     def dequeue(self):
#         if self.is_empty():
#             return "❌ Queue is empty."
#         return self.items.pop(0)

#     def peek(self):
#         if self.is_empty():
#             return "❌ Queue is empty."
#         return self.items[0]

#     def is_empty(self):
#         return len(self.items) == 0

#     def display(self):
#         print("🚌 Queue (front to back):")
#         for item in self.items:
#             print(item)

# q = Queue()

# q.enqueue("Customer1")
# q.enqueue("Customer2")
# q.enqueue("Customer3")
# q.enqueue("Customer4")
# q.enqueue("Customer5")
# q.enqueue("Customer6")
# q.enqueue("Customer7")
# q.enqueue("Customer8")

# q.display()

# print("Front of queue:", q.peek())
# print("🚪 Leaving:", q.dequeue())
# q.display()

# print("Front of queue:", q.peek())
# print("🚪 Leaving:", q.dequeue())
# q.display()

# print("Front of queue:", q.peek())
# print("🚪 Leaving:", q.dequeue())
# q.display()

# print("Front of queue:", q.peek())
# print("🚪 Leaving:", q.dequeue())
# q.display()

#-------------------------------------------------------------------------------

# class Queue:
#     def __init__(self):
#         self.items = []

#     def enqueue(self, item):
#         self.items.append(item)

#     def dequeue(self):
#         if not self.items:
#             return None
#         return self.items.pop(0)

#     def display(self):
#         if not self.items:
#             print("No normal patients.")
#         else:
#             print("🧑‍⚕️ Normal Queue:")
#             for i in self.items:
#                 print(f" - {i}")

#     def is_empty(self):
#         return len(self.items) == 0


# class Stack:
#     def __init__(self):
#         self.items = []

#     def push(self, item):
#         self.items.append(item)

#     def pop(self):
#         if not self.items:
#             return None
#         return self.items.pop()

#     def display(self):
#         if not self.items:
#             print("No emergency patients.")
#         else:
#             print("🚨 Emergency Stack:")
#             for i in reversed(self.items):
#                 print(f" - {i}")

#     def is_empty(self):
#         return len(self.items) == 0


# # Initialize both
# normal_queue = Queue()
# emergency_stack = Stack()

# # Menu
# while True:
#     print("\n===== 🏥 Hospital System =====")
#     print("1. Add Normal Patient")
#     print("2. Add Emergency Patient")
#     print("3. View All Patients")
#     print("4. Serve Next Patient")
#     print("5. Exit")

#     choice = input("Choose option: ")

#     if choice == '1':
#         name = input("Enter patient's name: ")
#         normal_queue.enqueue(name)
#         print("✅ Added to normal queue.")
#     elif choice == '2':
#         name = input("Enter emergency patient's name: ")
#         emergency_stack.push(name)
#         print("🚨 Emergency patient added.")
#     elif choice == '3':
#         emergency_stack.display()
#         normal_queue.display()
#     elif choice == '4':
#         if not emergency_stack.is_empty():
#             patient = emergency_stack.pop()
#             print(f"🚑 Serving Emergency Patient: {patient}")
#         elif not normal_queue.is_empty():
#             patient = normal_queue.dequeue()
#             print(f"🧑‍⚕️ Serving Normal Patient: {patient}")
#         else:
#             print("✅ No patients in line.")
#     elif choice == '5':
#         print("👋 Exiting system...")
#         break
#     else:
#         print("❌ Invalid option.")

#-------------------------------------------------------------------------------

# # A node has data + next pointer
# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.next = None

# # Linked list with basic operations
# class LinkedList:
#     def __init__(self):
#         self.head = None

#     def add(self, data):
#         new_node = Node(data)
#         if self.head is None:
#             self.head = new_node
#         else:
#             # traverse to the end
#             current = self.head
#             while current.next:
#                 current = current.next
#             current.next = new_node

#     def display(self):
#         current = self.head
#         if not current:
#             print("❌ List is empty.")
#             return

#         print("🔗 Linked List:")
#         while current:
#             print(f"[{current.data}] → ", end="")
#             current = current.next
#         print("None")

# ll = LinkedList()

# ll.add("Python")
# ll.add("Java")
# ll.add("C++")
# ll.display()
# ll.add("JavaScript")
# ll.display()

#-------------------------------------------------------------------------------

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.next = None

# class LinkedList:
#     def __init__(self):
#         self.head = None

#     # Add at end
#     def add(self, data):
#         new_node = Node(data)
#         if self.head is None:
#             self.head = new_node
#         else:
#             cur = self.head
#             while cur.next:
#                 cur = cur.next
#             cur.next = new_node

#     # Insert at position
#     def insert_at(self, pos, data):
#         new_node = Node(data)
#         if pos == 0:
#             new_node.next = self.head
#             self.head = new_node
#             return

#         cur = self.head
#         count = 0
#         while cur and count < pos - 1:
#             cur = cur.next
#             count += 1

#         if not cur:
#             print("❌ Position out of range")
#             return

#         new_node.next = cur.next
#         cur.next = new_node

#     # Delete by value
#     def delete(self, value):
#         cur = self.head
#         prev = None

#         while cur:
#             if cur.data == value:
#                 if prev:
#                     prev.next = cur.next
#                 else:
#                     self.head = cur.next
#                 print(f"🗑️ Deleted: {value}")
#                 return
#             prev = cur
#             cur = cur.next

#         print("❌ Value not found.")

#     # Search by value
#     def search(self, value):
#         cur = self.head
#         pos = 0
#         while cur:
#             if cur.data == value:
#                 print(f"🔍 Found '{value}' at position {pos}")
#                 return
#             cur = cur.next
#             pos += 1
#         print("❌ Not found.")

#     # Display
#     def display(self):
#         cur = self.head
#         if not cur:
#             print("❌ List is empty.")
#             return

#         print("🔗 Linked List:")
#         while cur:
#             print(f"[{cur.data}] → ", end="")
#             cur = cur.next
#         print("None")

# ll = LinkedList()

# ll.add("Python")
# ll.add("Java")
# ll.add("C++")
# ll.display()

# ll.insert_at(1, "Go")
# ll.display()

# ll.delete("Java")
# ll.display()

# ll.search("C++")
# ll.search("Rust")

#--------------------------------------------------------------------------------

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.prev = None
#         self.next = None


# class DoublyLinkedList:
#     def __init__(self):
#         self.head = None

#     def add_end(self, data):
#         new_node = Node(data)

#         if not self.head:
#             self.head = new_node
#             return

#         cur = self.head
#         while cur.next:
#             cur = cur.next

#         cur.next = new_node
#         new_node.prev = cur

#     def delete(self, data):
#         cur = self.head

#         while cur:
#             if cur.data == data:
#                 if cur.prev:
#                     cur.prev.next = cur.next
#                 else:
#                     self.head = cur.next

#                 if cur.next:
#                     cur.next.prev = cur.prev

#                 print(f"🗑️ Deleted: {data}")
#                 return
#             cur = cur.next

#         print("❌ Not found.")

#     def display_forward(self):
#         print("🔜 Forward Traversal:")
#         cur = self.head
#         while cur:
#             print(f"[{cur.data}]", end=" ⇄ ")
#             if cur.next is None:  # Save for reverse later
#                 self.tail = cur
#             cur = cur.next
#         print("None")

#     def display_backward(self):
#         print("🔙 Backward Traversal:")
#         cur = self.tail
#         while cur:
#             print(f"[{cur.data}]", end=" ⇄ ")
#             cur = cur.prev
#         print("None")

# dll = DoublyLinkedList()

# dll.add_end("HTML")
# dll.add_end("CSS")
# dll.add_end("JS")
# dll.add_end("Python")

# dll.display_forward()
# dll.display_backward()

# dll.delete("CSS")
# dll.display_forward()
# dll.display_backward()

#--------------------------------------------------------------------------------

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.next = None


# class Stack:
#     def __init__(self):
#         self.top = None

#     def push(self, data):
#         new_node = Node(data)
#         new_node.next = self.top  # new node points to current top
#         self.top = new_node       # update top

#     def pop(self):
#         if self.top is None:
#             return "❌ Stack is empty."
#         popped = self.top.data
#         self.top = self.top.next  # move top down
#         return popped

#     def peek(self):
#         if self.top is None:
#             return "❌ Stack is empty."
#         return self.top.data

#     def is_empty(self):
#         return self.top is None

#     def display(self):
#         if self.top is None:
#             print("❌ Stack is empty.")
#             return

#         print("📦 Stack (Top to Bottom):")
#         current = self.top
#         while current:
#             print(f"[{current.data}]")
#             current = current.next

# s = Stack()

# s.push("Undo")
# s.push("Redo")
# s.push("Save")

# s.display()
# print("👀 Peek:", s.peek())
# print("🗑️ Pop:", s.pop())
# s.display()

#--------------------------------------------------------------------------------

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.next = None


# class Queue:
#     def __init__(self):
#         self.front = None
#         self.rear = None

#     def enqueue(self, data):
#         new_node = Node(data)

#         if self.rear is None:  # Empty queue
#             self.front = self.rear = new_node
#         else:
#             self.rear.next = new_node
#             self.rear = new_node

#     def dequeue(self):
#         if self.front is None:
#             return "❌ Queue is empty."

#         removed = self.front.data
#         self.front = self.front.next

#         if self.front is None:  # Queue became empty
#             self.rear = None

#         return removed

#     def peek(self):
#         if self.front is None:
#             return "❌ Queue is empty."
#         return self.front.data

#     def is_empty(self):
#         return self.front is None

#     def display(self):
#         if self.front is None:
#             print("❌ Queue is empty.")
#             return

#         print("🚌 Queue (Front to Rear):")
#         temp = self.front
#         while temp:
#             print(f"[{temp.data}]", end=" -> ")
#             temp = temp.next
#         print("None")

# q = Queue()

# q.enqueue("Customer1")
# q.enqueue("Customer2")
# q.enqueue("Customer3")

# q.display()

# print("👀 Peek:", q.peek())
# print("🗑️ Dequeue:", q.dequeue())

# q.display()

#--------------------------------------------------------------------------------

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.next = None


# class Stack:
#     def __init__(self):
#         self.top = None

#     def push(self, data):
#         new = Node(data)
#         new.next = self.top
#         self.top = new

#     def pop(self):
#         if self.top is None:
#             return "❌ Stack is empty."
#         val = self.top.data
#         self.top = self.top.next
#         return val

#     def display(self):
#         print("📦 Stack (Top → Bottom):")
#         current = self.top
#         if not current:
#             print("Empty stack.")
#             return
#         while current:
#             print(f"[{current.data}]")
#             current = current.next


# class Queue:
#     def __init__(self):
#         self.front = None
#         self.rear = None

#     def enqueue(self, data):
#         new = Node(data)
#         if self.rear is None:
#             self.front = self.rear = new
#         else:
#             self.rear.next = new
#             self.rear = new

#     def dequeue(self):
#         if self.front is None:
#             return "❌ Queue is empty."
#         val = self.front.data
#         self.front = self.front.next
#         if self.front is None:
#             self.rear = None
#         return val

#     def display(self):
#         print("🚌 Queue (Front → Rear):")
#         current = self.front
#         if not current:
#             print("Empty queue.")
#             return
#         while current:
#             print(f"[{current.data}] → ", end="")
#             current = current.next
#         print("None")

# stack = Stack()
# queue = Queue()

# while True:
#     print("\n===== Stack & Queue Simulator =====")
#     print("1. Use Stack")
#     print("2. Use Queue")
#     print("3. Exit")

#     choice = input("Choose option: ")

#     if choice == "1":
#         while True:
#             print("\n--- Stack Menu ---")
#             print("a. Push")
#             print("b. Pop")
#             print("c. Display")
#             print("d. Back")

#             op = input("Enter choice: ").lower()

#             if op == "a":
#                 val = input("Enter value to push: ")
#                 stack.push(val)
#                 print("✅ Pushed to stack.")
#             elif op == "b":
#                 print("🗑️ Popped:", stack.pop())
#             elif op == "c":
#                 stack.display()
#             elif op == "d":
#                 break
#             else:
#                 print("❌ Invalid option.")

#     elif choice == "2":
#         while True:
#             print("\n--- Queue Menu ---")
#             print("a. Enqueue")
#             print("b. Dequeue")
#             print("c. Display")
#             print("d. Back")

#             op = input("Enter choice: ").lower()

#             if op == "a":
#                 val = input("Enter value to enqueue: ")
#                 queue.enqueue(val)
#                 print("✅ Enqueued to queue.")
#             elif op == "b":
#                 print("🗑️ Dequeued:", queue.dequeue())
#             elif op == "c":
#                 queue.display()
#             elif op == "d":
#                 break
#             else:
#                 print("❌ Invalid option.")

#     elif choice == "3":
#         print("👋 Exiting...")
#         break
#     else:
#         print("❌ Invalid choice.")

#--------------------------------------------------------------------------------

#🌳 Trees 

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.left = None
#         self.right = None


# class BinaryTree:
#     def __init__(self):
#         self.root = None

#     def insert(self, data):
#         if self.root is None:
#             self.root = Node(data)
#         else:
#             self._insert_recursive(self.root, data)

#     def _insert_recursive(self, current, data):
#         if data < current.data:
#             if current.left is None:
#                 current.left = Node(data)
#             else:
#                 self._insert_recursive(current.left, data)
#         else:
#             if current.right is None:
#                 current.right = Node(data)
#             else:
#                 self._insert_recursive(current.right, data)

#     def inorder(self):
#         print("🔁 Inorder Traversal (Left → Root → Right):")
#         self._inorder_recursive(self.root)

#     def _inorder_recursive(self, node):
#         if node:
#             self._inorder_recursive(node.left)
#             print(node.data, end=" ")
#             self._inorder_recursive(node.right)


# bt = BinaryTree()

# bt.insert(10)
# bt.insert(5)
# bt.insert(15)
# bt.insert(2)
# bt.insert(7)
# bt.insert(20)

# bt.inorder()
# bt = BinaryTree()

# bt.insert(10)
# bt.insert(5)
# bt.insert(15)
# bt.insert(2)
# bt.insert(7)
# bt.insert(20)

# bt.inorder()

#--------------------------------------------------------------------------------

# class Node:
#     def __init__(self, name, score):
#         self.name = name
#         self.score = score
#         self.left = None
#         self.right = None


# class StudentTree:
#     def __init__(self):
#         self.root = None

#     def insert(self, name, score):
#         new_node = Node(name, score)
#         if self.root is None:
#             self.root = new_node
#         else:
#             self._insert_recursive(self.root, new_node)

#     def _insert_recursive(self, current, new_node):
#         if new_node.score < current.score:
#             if current.left is None:
#                 current.left = new_node
#             else:
#                 self._insert_recursive(current.left, new_node)
#         else:
#             if current.right is None:
#                 current.right = new_node
#             else:
#                 self._insert_recursive(current.right, new_node)

#     def inorder(self):
#         print("🔁 Inorder (Left → Root → Right):")
#         self._inorder_recursive(self.root)
#         print()

#     def _inorder_recursive(self, node):
#         if node:
#             self._inorder_recursive(node.left)
#             print(f"{node.name} ({node.score})", end="  ")
#             self._inorder_recursive(node.right)

#     def preorder(self):
#         print("▶️ Preorder (Root → Left → Right):")
#         self._preorder_recursive(self.root)
#         print()

#     def _preorder_recursive(self, node):
#         if node:
#             print(f"{node.name} ({node.score})", end="  ")
#             self._preorder_recursive(node.left)
#             self._preorder_recursive(node.right)

#     def postorder(self):
#         print("⏹️ Postorder (Left → Right → Root):")
#         self._postorder_recursive(self.root)
#         print()

#     def _postorder_recursive(self, node):
#         if node:
#             self._postorder_recursive(node.left)
#             self._postorder_recursive(node.right)
#             print(f"{node.name} ({node.score})", end="  ")

#     def search(self, score):
#         return self._search_recursive(self.root, score)

#     def _search_recursive(self, node, score):
#         if node is None:
#             return None
#         if node.score == score:
#             return node
#         elif score < node.score:
#             return self._search_recursive(node.left, score)
#         else:
#             return self._search_recursive(node.right, score)

#     def find_min(self):
#         current = self.root
#         while current and current.left:
#             current = current.left
#         return current

#     def find_max(self):
#         current = self.root
#         while current and current.right:
#             current = current.right
#         return current

#     def height(self):
#         return self._height_recursive(self.root)

#     def _height_recursive(self, node):
#         if node is None:
#             return -1
#         return 1 + max(self._height_recursive(node.left),
#                        self._height_recursive(node.right))

# tree = StudentTree()

# # Add students
# tree.insert("Saadi", 88)
# tree.insert("Ali", 72)
# tree.insert("Sara", 95)
# tree.insert("Babar", 66)
# tree.insert("Zoya", 90)
# tree.insert("Umar", 80)

# tree.inorder()
# tree.preorder()
# tree.postorder()

# # Search for a score
# target = 90
# found = tree.search(target)
# if found:
#     print(f"\n🎯 Found: {found.name} with score {found.score}")
# else:
#     print(f"\n❌ Score {target} not found")

# # Find min and max
# min_node = tree.find_min()
# max_node = tree.find_max()
# print(f"\n🏆 Min Score: {min_node.name} ({min_node.score})")
# print(f"🥇 Max Score: {max_node.name} ({max_node.score})")

# # Height of the tree
# print(f"\n🌲 Height of Tree: {tree.height()}")

#-------------------------------------------------------------------------------

# from collections import defaultdict, deque

# class Graph:
#     def __init__(self):
#         self.graph = defaultdict(list)  # adjacency list

#     def add_edge(self, u, v, directed=False):
#         self.graph[u].append(v)
#         if not directed:
#             self.graph[v].append(u)

#     def display(self):
#         print("\n🌐 Graph Adjacency List:")
#         for node in self.graph:
#             print(f"{node} ➡️ {self.graph[node]}")

#     def bfs(self, start):
#         visited = set()
#         queue = deque([start])
#         print(f"\n🔍 BFS starting from {start}: ", end="")

#         while queue:
#             node = queue.popleft()
#             if node not in visited:
#                 print(node, end=" → ")
#                 visited.add(node)
#                 queue.extend(self.graph[node])

#     def dfs(self, start):
#         visited = set()
#         print(f"\n🧭 DFS starting from {start}: ", end="")
#         self._dfs_recursive(start, visited)

#     def _dfs_recursive(self, node, visited):
#         if node not in visited:
#             print(node, end=" → ")
#             visited.add(node)
#             for neighbor in self.graph[node]:
#                 self._dfs_recursive(neighbor, visited)

# g = Graph()

# # Add nodes and connections (undirected)
# g.add_edge("Saadi", "Ali")
# g.add_edge("Ali", "Babar")
# g.add_edge("Saadi", "Zoya")
# g.add_edge("Zoya", "Sara")
# g.add_edge("Sara", "Umar")
# g.add_edge("Ali", "Umar")

# g.display()

# # Traversals
# g.bfs("Saadi")
# g.dfs("Saadi")

#-------------------------------------------------------------------------------

