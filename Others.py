# class Bank:
#     accounts_created = 0  # Class variable

#     def __init__(self, name):
#         self.name = name
#         Bank.accounts_created += 1

#     @classmethod
#     def how_many_accounts(cls):
#         print(f"👥 Total accounts created: {cls.accounts_created}")

# b1 = Bank("Bank A")
# b2 = Bank("Bank B") 
# b3 = Bank("Bank C")
# Bank.how_many_accounts()  # Output: 👥 Total accounts created: 2





# from math_tools import square, is_even

# print("Square of 4 is:", square(4))
# print("Is 4 even?", is_even(4))
# print("Is 5 even?", is_even(5))







# def square(n):
#     """Returns the square of x."""
#     return n * n

# def is_even(n):
#     """Returns True if n is even, False otherwise."""
#     return n % 2 == 0









# # utils.py

# def greet(name):
#     return f"Hello, {name}! 👋"

# def greet2(name):
#     return f"Hello, {name}! 👋"

# def add(a, b):
#     return a + b

# def subtract(a, b):
#     return a - b

# def multiply(a, b):
#     return a * b    

# def divide(a, b):
#     if b == 0:
#         return "Error: Division by zero is not allowed."
#     return a / b    













# class Person:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age

#     def display_info(self):
#         print(f"👤 Name: {self.name}, Age: {self.age}")

# class Student(Person):
#     def __init__(self, name, age, student_id):
#         super().__init__(name, age)
#         self.student_id = student_id
#         self.courses = []

#     def enroll(self, course_name):
#         self.courses.append(course_name)
#         print(f"✅ {self.name} enrolled in {course_name}")

#     def display_info(self):
#         super().display_info()
#         print(f"🎓 Student ID: {self.student_id}")
#         print("📚 Enrolled Courses:", ", ".join(self.courses))

# class Teacher(Person):
#     def __init__(self, name, age, subject):
#         super().__init__(name, age)
#         self.subject = subject

#     def display_info(self):
#         super().display_info()
#         print(f"📘 Teaches: {self.subject}")

# class Course:
#     def __init__(self, course_name, teacher):
#         self.course_name = course_name
#         self.teacher = teacher
#         self.students = []

#     def add_student(self, student):
#         self.students.append(student)
#         student.enroll(self.course_name)

#     def display_course(self):
#         print(f"📖 Course: {self.course_name}")
#         print(f"👨‍🏫 Teacher: {self.teacher.name}")
#         print("👥 Students Enrolled:")
#         for s in self.students:
#             print(f" - {s.name}")


# # Create teacher and student
# t1 = Teacher("Sir Ali", 35, "Python")
# t2 = Teacher("Ms. Fatima", 30, "JavaScript")
# s1 = Student("Saadi", 19, "S123")
# s2 = Student("Umer", 20, "S456")
# s3 = Student("Sara", 18, "S789")

# # Create course
# python_course = Course("Python Programming", t1)
# javascript_course = Course("JavaScript Programming", t2)

# # Enroll students
# python_course.add_student(s1)
# python_course.add_student(s2)
# javascript_course.add_student(s3)
# javascript_course.add_student(s1)  

# # Show details
# print("\n--- Teacher ---")
# t1.display_info()
# t2.display_info()

# print("\n--- Students ---")
# s1.display_info()
# s2.display_info()
# s3.display_info()  

# print("\n--- Course ---")
# python_course.display_course()
# javascript_course.display_course()









# class Car:
#     def __init__(self, brand, model, year):
#         self.brand = brand
#         self.model = model
#         self.year = year
    
#     def display(self):
#         print(f"This is a {self.year} {self.brand} {self.model}.")


# c1 = Car("Toyota", "Corolla", 2020)
# c2 = Car("Honda", "Civic", 2021)
# c3 = Car("Ford", "Mustang", 2022)
# c4 = Car("Tesla", "Model 3", 2023)

# c1.display()
# c2.display()
# c3.display()
# c4.display()











# # main.py
# from utils import greet, add, subtract, multiply, divide, greet2

# print("Sum is:",add(5, 3))
# print("Difference is:",subtract(10, 4))
# print("Product is:",multiply(6, 7))
# print("Quotient is:",divide(20, 4))
# print(greet("Ali"))  # No need to write utils.greet
# print(greet2("Sara"))  # No need to write utils.greet








# import random

# # List of questions
# questions = [
#     {"question": "What is the capital of Pakistan?", "answer": "Islamabad"},
#     {"question": "What is 5 + 7?", "answer": "12"},
#     {"question": "What programming language are we learning?", "answer": "Python"},
#     {"question": "Which planet is known as the Red Planet?", "answer": "Mars"}
# ]

# # Pick a random question
# q = random.choice(questions)

# # Ask the user
# print("🧠 Quiz Time!")
# user_answer = input(q["question"] + " ")

# # Check answer
# if user_answer.strip().lower() == q["answer"].lower():
#     print("✅ Correct!")
# else:
#     print(f"❌ Wrong. The correct answer is {q['answer']}.")
    
          
# print(random.choice(["apple", "banana", "mango"]))  # Random choice

# import datetime

# today = datetime.date.today()
# print("Today's date is:", today)



# class BankAccount:
#     def __init__(self, name, balance=0):
#         self.name = name
#         self.balance = balance

#     def show_balance(self):
#         print(f"{self.name}'s balance is {self.balance}")

#     def deposit(self, amount):
#         self.balance += amount
#         print(f"💰 Deposited {amount}. New balance: {self.balance}")

#     def withdraw(self, amount):
#         if amount > self.balance:
#             print("❌ Insufficient balance!")
#         else:
#             self.balance -= amount
#             print(f"💸 Withdrawn {amount}. New balance: {self.balance}")

# class SavingsAccount(BankAccount):
#     def __init__(self, name, balance=0, interest_rate=0.02):
#         super().__init__(name, balance)  # Inherit parent's __init__
#         self.interest_rate = interest_rate

#     def apply_interest(self):
#         interest = self.balance * self.interest_rate
#         self.balance += interest
#         print(f"💹 Interest of {interest} added. New balance: {self.balance}")

# sa = SavingsAccount("Saadi", 1000)
# sa.show_balance()
# sa.apply_interest()
# sa.withdraw(500)
# sa.show_balance()
# sa.deposit(200)
# sa.show_balance()
# sa.withdraw(800)  # Should show insufficient balance
# sa.show_balance()
# sa.apply_interest()  # Apply interest again
# sa.show_balance()                   











# file = open("data.txt", "w")   # "w" = write (overwrites)
# file.write("Hello Saadi!")
# file.close()


# file = open("data.txt", "r")
# content = file.read()
# print(content)
# file.close()


# with open("data.txt", "w") as file:
#     file.write("This is saved safely!")


# with open("data.txt", "r") as file:
#     content = file.read()
#     print(content)


# note = input("📝 Enter a note to save: ")
# with open("data.txt", "a") as file:
#     file.write(note + "\n")
# print("✅ Note saved to data.txt")


# from datetime import datetime
# note = input("📝 Enter a note to append: ")
# time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# with open("data.txt", "a") as file:
#     file.write(f"[{time}] {note}\n")
# print("✅ Note with timestamp appended to data.txt")


# while True:
#     print("\nNotes Menu:")
#     print("1. Read Notes")
#     print("2. Write to Notes")
#     print("3. Append to Notes")
#     print("4. Clear All Notes")
#     print("5. Exit")
#     choice = input("Choose an option (1-5): ")

#     if choice == '1':
#         with open("data.txt", "r") as file:
#             content = file.read()
#             print("\nCurrent Notes:")
#             print(content if content else "No notes found.")
    
#     elif choice == '2':
#         note = input("📝 Enter a note to write (overwrites existing content): ")
#         with open("data.txt", "w") as file:
#             file.write(note + "\n")
#         print("✅ Note written to data.txt")
    
#     elif choice == '3':
#         note = input("📝 Enter a note to append: ")
#         with open("data.txt", "a") as file:
#             file.write(note + "\n")
#         print("✅ Note appended to data.txt")
    
#     elif choice == '4':
#         with open("data.txt", "w") as file:
#             file.write("")  # Clears the file
#         print("🗑️ All notes cleared.")

#     elif choice == '5':
#         print("Exiting the Notes Menu. Goodbye!")
#         break
#     else:
#         print("❌ Invalid choice. Please select a valid option (1-4).") 











# # Can you check anyone Location using Python Code

# import phonenumbers
# from phonenumbers import timezone
# from phonenumbers import geocoder
# from phonenumbers import carrier

# # Enter phone number along with country code
# number = input("Enter phone number with country code : ")

# # Parsing String to the Phone number
# phoneNumber = phonenumbers.parse(number)

# # printing the timezone using the timezone module
# timeZone = timezone.time_zones_for_number(phoneNumber)
# print("timeZone : " + str(timeZone))

# # printing the geolocation of the given number using the geocoder module
# geolocation = geocoder.description_for_number(phoneNumber, "en")
# print("location : " + geolocation)

# # printing the service provider
# service = carrier.name_for_number(phoneNumber, "en")
# print("service provider : " + service)








