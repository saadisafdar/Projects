fruits = ["apple", "banana", "cherry", "date", "mango"]
for f in fruits:
    print(f)
    if f == "cherry":
        break

s = "programming"
for c in s:
    if c == "m":
        continue
    print(c)

# This code encrypts and decrypts a message using a simple shift method

message = "Hello World"
key = 2  # shifting each letter by 2

# Encrypt the message
encrypted = ""
for char in message:
    encrypted += chr(ord(char) + key)

print("Encrypted Message:", encrypted)

# Decrypt the message
decrypted = ""
for char in encrypted:
    decrypted += chr(ord(char) - key)

print("Decrypted Message:", decrypted)

# This code converts a password into a hashed form using hashlib

import hashlib

password = input("Enter password: ")
hashed = hashlib.sha256(password.encode()).hexdigest()

print("Hashed password:", hashed)

# This code checks if two files are the same or changed

import hashlib

def get_hash(filename):
    with open(filename, "rb") as f:
        data = f.read()
        return hashlib.sha256(data).hexdigest()

file1 = "file1.txt"
file2 = "file2.txt"

if get_hash(file1) == get_hash(file2):
    print("Files are same (No change found)")
else:
    print("Files are different (File may be modified)")

# This code checks if an IP address is allowed or blocked

blocked_ips = ["192.168.1.10", "10.0.0.5"]

ip = input("Enter IP address: ")

if ip in blocked_ips:
    print("Access Denied! IP is blocked.")
else:
    print("Access Granted.")

# This code finds any suspicious process names

processes = ["chrome.exe", "python.exe", "virus.exe"]

for p in processes:
    if "virus" in p:
        print("Suspicious process found:", p)
    else:
        print("Safe process:", p)

# This code checks password and a random OTP
import random
password = "admin123"
user_pass = input("Enter password: ")
if user_pass == password:
    otp = random.randint(1000, 9999)
    print("Your OTP is:", otp)
    entered_otp = int(input("Enter OTP: "))
    if entered_otp == otp:
        print("Login Successful!")
    else:
        print("Incorrect OTP!")
else:
    print("Wrong Password!")

# This code encrypts and decrypts text using Caesar Cipher method

def caesar_cipher(text, key):
    result = ""
    for char in text:
        result += chr(ord(char) + key)
    return result

message = "Cyber Security"
key = 3

encrypted = caesar_cipher(message, key)
print("Encrypted:", encrypted)

decrypted = caesar_cipher(encrypted, -key)
print("Decrypted:", decrypted)



# Lab Assignment

# Task 1:
# 1. Write a program that takes a list of student IDs (integers) entered by the user.
# 2. Use a for loop to count how many IDs were scanned.
# 3. If any ID appears more than once, display a warning message for that student.
# 4. Stop scanning if the ID entered is 0 (use break).
# 5. Finally, print total students present.
# Hint: Use loops and if conditions to detect duplicates.

# Task 2:
# 1. Take a password input from the user.
# 2. Use a while loop to re-ask the user until the password is at least 8 characters long.
# 3. Then use a for loop to check:
# •Must contain at least one uppercase letter
# •Must contain at least one number
# • Must contain one special character (!@#$%&*)
# 4. If any condition fails, show a hint to the user to improve the password.
# Hint: Use any() and string methods like .isupper() and .isdigit().

# Task 3:
# 1. Create a function add_student(name, *marks) that stores student data (name and marks)
# in a list.
# 2. Create another function calculate_average(*marks) that returns the average marks.
# 3. Create a third function assign_grade(average) that returns grades based on the scale.
# 4. Allow the teacher to enter data for 5 students using loops.
# 5. Display all students’ names with their averages and grades in a formatted way.
# Hint: Combine loops, functions, and *args for flexibility.

# Task 4:
# 1. Take 7 temperature readings (integers) for a week and store them in a list.
# 2. Write a function get_max_min(temps) that returns both maximum and minimum
# temperatures.
# 3. Write another function above_average(temps) that prints all days with above-average
# temperature.
# 4. Do not use built-in functions like max() or min() — calculate manually using loops.
# Hint: Calculate total, divide by length, and compare each element.

# Task 5:
# 1. Create a class Book with attributes: title, author, and is_borrowed.
# 2. Use a constructor to initialize these.
# 3. Create methods:
# •borrow_book() → marks the book as borrowed
# •return_book() → marks the book as available
# • display_info() → shows book details and status
# 4. In main, create a list of 5 books, let user borrow or return books using input and loops.
# Hint: Manage book status using Boolean values.

# Task 6:
# 1. Create a class Employee with attributes: name, department, and monthly_scores (list).
# 2. Create methods:
# •add_score(score) → adds monthly score to the list
# •average_score() → calculates and returns average
# •performance_level() → returns “Excellent”, “Good”, “Average”, or “Poor”
# based on average
# 3. In main:
# •Create 3 employee objects
# •Add random monthly scores using a loop
# • Display each employee’s average and performance level
# Hint: Use list operations and class methods together.




# Task 1
id = []
while True:
    student_id = int(input("Enter student ID (0 to stop): "))
    
    if student_id == 0:
        break
    
    if student_id in id:
        print("Warning: Student ID ", student_id, " is a duplicate!")
    
    else:
        id.append(student_id)

print("Total students present: ", len(id))

#-------------------------------------------------------------------------------------------------

# Task 2
special_chars = "!@#$%&*"

while True:
    password = input("Enter a password (at least 8 characters): ")

    if len(password) < 8:
        print("Password must be at least 8 characters long.")
        continue
    if not any(c.isupper() for c in password):
        print("Password must contain at least one uppercase letter.")
        continue
    if not any(c.isdigit() for c in password):
        print("Password must contain at least one number.")
        continue
    if not any(c in special_chars for c in password):
        print("Password must contain at least one special character (!@#$%&*).")
        continue

    print("Password is valid.")
    break


#-------------------------------------------------------------------------------------------------

# Task 3
def add_student(name, *marks):
    return (name, marks)

def calculate_average(*marks):
    total = 0
    for m in marks:
        total += m
    return total / len(marks)

def assign_grade(average):
    if average >= 90:
        return "A"
    elif average >= 80:
        return "B"
    elif average >= 70:
        return "C"
    elif average >= 60:
        return "D"
    else:
        return "F"

students = []

for i in range(5):
    name = input("Enter student name: ")
    marks = []
    for j in range(3):
        mark = float(input("Enter mark {}: ".format(j+1)))
        marks.append(mark)
    students.append(add_student(name, *marks))

for student in students:
    name, marks = student
    avg = calculate_average(*marks)
    grade = assign_grade(avg)
    print("Name:", name, "| Average:", avg, "| Grade:", grade)

#-------------------------------------------------------------------------------------------------

# Task 4
temps = []

for i in range(7):
    t = int(input("Enter temperature for day {}: ".format(i+1)))
    temps.append(t)

def get_max_min(temps):
    max_t = temps[0]
    min_t = temps[0]
    for t in temps:
        if t > max_t:
            max_t = t
        if t < min_t:
            min_t = t
    return max_t, min_t

def above_average(temps):
    total = 0
    for t in temps:
        total += t
    avg = total / len(temps)
    print("Days with above-average temperature:")
    for i in range(len(temps)):
        if temps[i] > avg:
            print("Day", i+1, ":", temps[i])

max_t, min_t = get_max_min(temps)
print("Maximum temperature:", max_t)
print("Minimum temperature:", min_t)
above_average(temps)

#-------------------------------------------------------------------------------------------------
        
# Task 5
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author
        self.is_borrowed = False

    def borrow_book(self):
        if not self.is_borrowed:
            self.is_borrowed = True
            print("You borrowed:", self.title)
        else:
            print(self.title, "is already borrowed.")

    def return_book(self):
        if self.is_borrowed:
            self.is_borrowed = False
            print("You returned:", self.title)
        else:
            print(self.title, "was not borrowed.")

    def display_info(self):
        status = "Borrowed" if self.is_borrowed else "Available"
        print("Title:", self.title, "| Author:", self.author, "| Status:", status)

books = [
    Book("Jangloos", "Shaukat Siddiqui"),
    Book("Udaas Naslain", "Abdullah Hussain"),
    Book("Aangan", "Khadija Mastoor"),
    Book("Zavia", "Ashfaq Ahmed"),
    Book("Peer-e-Kamil", "Umera Ahmed")
]

for book in books:
    book.display_info()

action = input("Do you want to borrow or return a book? (b/r): ")
title = input("Enter book title: ")

for book in books:
    if book.title == title:
        if action == "b":
            book.borrow_book()
        elif action == "r":
            book.return_book()

#-------------------------------------------------------------------------------------------------

# Task 6
import random

class Employee:
    def __init__(self, name, department):
        self.name = name
        self.department = department
        self.monthly_scores = []

    def add_score(self, score):
        self.monthly_scores.append(score)

    def average_score(self):
        total = 0
        for s in self.monthly_scores:
            total += s
        return total / len(self.monthly_scores)

    def performance_level(self):
        avg = self.average_score()
        if avg >= 90:
            return "Excellent"
        elif avg >= 75:
            return "Good"
        elif avg >= 60:
            return "Average"
        else:
            return "Poor"

# Pakistani names and departments
employees = [
    Employee("Ahmed Khan", "Human Resources"),
    Employee("Fatima Malik", "Information Technology"),
    Employee("Bilal Ahmed", "Finance")
]

for emp in employees:
    for i in range(12):
        emp.add_score(random.randint(50, 100))

    avg = emp.average_score()
    perf = emp.performance_level()
    print("Name:", emp.name, "| Department:", emp.department, "| Average Score:", avg, "| Performance:", perf)
