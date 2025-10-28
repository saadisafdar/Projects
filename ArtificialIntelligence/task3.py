import math

# for i in range(1,6):
#     print("Hello World")

# for letter in "Saadi":
#     print(letter)   

# for number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
#     print(number)

# count = 1
# while count <= 5:
#     print("Count is:", count)
#     count = count + 1

# e = 1
# while e <= 20:
#     if e % 2 == 0:
#         print(e)
#     e += 1

# for num in range(1, 11):
#     if num % 2 == 0:
#         print("Even number:", num)
#     else:
#         print("Odd number:", num) 

# def greet():
#     print("Hello World")
# greet()

# def greet_user(name):
#     print("Hello ",name)
# greet_user("Saadi")

# def add_nums(a,b):
#     sum = a + b
#     print("The sum is: ",sum)
# add_nums(5,15)

# def multiply(a,b):
#     result = a * b
#     return result
# product = multiply(4,6)
# print("The product is: ", product)

# print(math.sqrt(16))
# print(math.pow(2, 3))
# print(math.ceil(5.5))
# print(math.floor(5.5))
# print(math.factorial(5))
# print(math.fabs(-10))
# print(math.gcd(48, 18))

# Lab 3 AI Tasks
# Q1. Write a Python program that takes a number from the user and prints its multiplication
# table from 1 to 10 using a for loop.
# Q2. Write a Python program using a while loop to calculate and print the sum of all even
# numbers between 1 and 50.
# Q3. Write a Python function named find_largest(a, b, c) that takes three numbers as input
# and prints the largest number among them.
# Q4. Write a Python function named calculate_area(radius) that takes the radius of a circle as
# input and returns the area of the circle. Use the formula: Area = 3.14 × radius2.
# Q5. Write a Python program that defines a nested function where the main function
# student_report() contains another function grade_status(marks). The inner function should
# print “Pass” if marks ≥ 50, otherwise print “Fail”. Call the inner function from within the
# main function.


# n = int(input("Enter a number to print its multiplication table: "))
# for i in range(1, 11):
#     print(n," x ",i," = ",n * i)

# p = 1
# sum = 0
# while p<=50:
#     if (p%2==0):
#         sum = sum + p
#     p = p + 1
# print("Sum of all even numbers between 1 and 50 is", sum)

# def find_largest(a, b, c):
#     if a >= b and a >= c:
#         print("The largest number is:", a)
#     elif b >= a and b >= c:
#         print("The largest number is:", b)
#     else:
#         print("The largest number is:", c)

# a = int(input("Enter first number: "))
# b = int(input("Enter second number: "))
# c = int(input("Enter third number: "))
# find_largest(a, b, c)

# def calculate_area(radius):
#     area = 3.14 * radius * radius
#     return area
# radius = float(input("Enter the radius of the circle: "))
# area = calculate_area(radius)
# print("The area of the circle  is: ", area)


# def student_report():
#     marks = float(input("Enter the marks: "))
#     def grade_status(marks):
#         if marks >= 50:
#             print("Pass")
#         else:
#             print("Fail")
#     grade_status(marks)
# student_report() 


# for i in range(1,6):
#     for j in range(1,6):
#         print(i, j)
    

# # Creating lists
# fruits = ["apple", "banana", "orange", "grape"]
# numbers = [1, 2, 3, 4, 5]
# mixed = [1, "hello", 3.14, True]

# print("Fruits:", fruits)
# print("Numbers:", numbers)
# print("Mixed:", mixed)

# # Accessing elements
# print("First fruit:", fruits[0])      # apple
# print("Last fruit:", fruits[-1])     # grape
# print("Slice:", fruits[1:3])         # ['banana', 'orange']

# # Modifying lists
# fruits.append("mango")               # Add to end
# print("After append:", fruits)       # ['apple', 'banana', 'orange', 'grape', 'mango']

# fruits.insert(1, "kiwi")             # Insert at position
# print("After insert:", fruits)       # ['apple', 'kiwi', 'banana', 'orange', 'grape', 'mango']

# fruits.remove("banana")              # Remove element
# print("After remove:", fruits)       # ['apple', 'kiwi', 'orange', 'grape', 'mango']

# popped = fruits.pop()                # Remove last element
# print("Popped:", popped)             # mango
# print("After pop:", fruits)          # ['apple', 'kiwi', 'orange', 'grape']

# # List operations
# fruits.extend(["strawberry", "blueberry"])
# print("After extend:", fruits)

# fruits.sort()
# print("Sorted:", fruits)

# print("Length:", len(fruits))
# print("Is 'apple' in list?", "apple" in fruits)



# # Creating tuples
# colors = ("red", "green", "blue")
# coordinates = (10, 20)
# single_item = (42,)  # Note: comma is required for single item
# mixed_tuple = (1, "hello", 3.14)

# print("Colors:", colors)
# print("Coordinates:", coordinates)
# print("Single item:", single_item)

# # Accessing elements (same as lists)
# print("First color:", colors[0])     # red
# print("Last color:", colors[-1])     # blue
# print("Slice:", colors[1:])          # ('green', 'blue')

# # Tuples are immutable - these would cause errors:
# # colors[0] = "yellow"  # TypeError!
# # colors.append("black") # AttributeError!

# # Tuple unpacking
# x, y = coordinates
# print(f"x: {x}, y: {y}")  # x: 10, y: 20

# # Multiple return values from function
# def get_user_info():
#     return "John", 25, "john@email.com"

# name, age, email = get_user_info()
# print(f"Name: {name}, Age: {age}, Email: {email}")

# # Using tuples as dictionary keys
# location_keys = {
#     (35.6895, 139.6917): "Tokyo",
#     (40.7128, -74.0060): "New York"
# }
# print("Tokyo coordinates:", location_keys[(35.6895, 139.6917)])



# # Creating dictionaries
# student = {
#     "name": "Alice",
#     "age": 20,
#     "major": "Computer Science",
#     "grades": [85, 92, 78]
# }

# car = {
#     "brand": "Toyota",
#     "model": "Camry", 
#     "year": 2022,
#     "color": "blue"
# }

# print("Student:", student)
# print("Car:", car)

# # Accessing values
# print("Student name:", student["name"])           # Alice
# print("Student age:", student.get("age"))         # 20
# print("Unknown key:", student.get("height", "Not specified"))  # Default value

# # Adding and modifying
# student["gpa"] = 3.8              # Add new key-value
# student["age"] = 21               # Modify existing value
# car["color"] = "red"              # Change value

# print("Updated student:", student)
# print("Updated car:", car)

# # Removing elements
# removed_grade = student.pop("grades")  # Remove and return value
# print("Removed grades:", removed_grade)
# print("After pop:", student)

# del car["year"]                    # Remove key-value pair
# print("After del:", car)

# # Dictionary methods
# print("Keys:", student.keys())     # All keys
# print("Values:", student.values()) # All values
# print("Items:", student.items())   # Key-value pairs

# # Iterating through dictionary
# print("\nStudent information:")
# for key, value in student.items():
#     print(f"{key}: {value}")

# # Checking existence
# print("Has name?", "name" in student)     # True
# print("Has height?", "height" in student) # False



# # List of dictionaries
# students = [
#     {"name": "Alice", "age": 20, "grade": "A"},
#     {"name": "Bob", "age": 22, "grade": "B"},
#     {"name": "Charlie", "age": 21, "grade": "A"}
# ]

# print("Students list:")
# for student in students:
#     print(f"- {student['name']}, Age: {student['age']}, Grade: {student['grade']}")

# # Dictionary with list values
# classroom = {
#     "math": ["Alice", "Bob", "Charlie"],
#     "science": ["David", "Eve"],
#     "history": ["Frank", "Grace", "Heidi"]
# }

# print("\nClassroom subjects:")
# for subject, students_list in classroom.items():
#     print(f"{subject}: {', '.join(students_list)}")

# # Tuple unpacking with lists
# coordinates_list = [(1, 2), (3, 4), (5, 6)]
# print("\nCoordinates:")
# for x, y in coordinates_list:
#     print(f"x: {x}, y: {y}")

# # Simple inventory system
# inventory = {
#     "apples": 10,
#     "bananas": 15,
#     "oranges": 8
# }

# # Add items
# inventory["grapes"] = 12

# # Update quantity
# inventory["apples"] += 5

# print("\nInventory:")
# for item, quantity in inventory.items():
#     print(f"{item}: {quantity}")

# # Check if item needs restocking
# for item, quantity in inventory.items():
#     if quantity < 10:
#         print(f"⚠️  Restock needed for {item}")