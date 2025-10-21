import math

for i in range(1,6):
    print("Hello World")

for letter in "Saadi":
    print(letter)   

for number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    print(number)

count = 1
while count <= 5:
    print("Count is:", count)
    count = count + 1

e = 1
while e <= 20:
    if e % 2 == 0:
        print(e)
    e += 1

for num in range(1, 11):
    if num % 2 == 0:
        print("Even number:", num)
    else:
        print("Odd number:", num) 

def greet():
    print("Hello World")
greet()

def greet_user(name):
    print("Hello ",name)
greet_user("Saadi")

def add_nums(a,b):
    sum = a + b
    print("The sum is: ",sum)
add_nums(5,15)

def multiply(a,b):
    result = a * b
    return result
product = multiply(4,6)
print("The product is: ", product)

print(math.sqrt(16))
print(math.pow(2, 3))
print(math.ceil(5.5))
print(math.floor(5.5))
print(math.factorial(5))
print(math.fabs(-10))
print(math.gcd(48, 18))

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


n = int(input("Enter a number to print its multiplication table: "))
for i in range(1, 11):
    print(n," x ",i," = ",n * i)

p = 1
sum = 0
while p<=50:
    if (p%2==0):
        sum = sum + p
    p = p + 1
print("Sum of all even numbers between 1 and 50 is", sum)

def find_largest(a, b, c):
    if a >= b and a >= c:
        print("The largest number is:", a)
    elif b >= a and b >= c:
        print("The largest number is:", b)
    else:
        print("The largest number is:", c)

a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
c = int(input("Enter third number: "))
find_largest(a, b, c)

def calculate_area(radius):
    area = 3.14 * radius * radius
    return area
radius = float(input("Enter the radius of the circle: "))
area = calculate_area(radius)
print("The area of the circle  is: ", area)


def student_report():
    marks = float(input("Enter the marks: "))
    def grade_status(marks):
        if marks >= 50:
            print("Pass")
        else:
            print("Fail")
    grade_status(marks)
student_report() 


for i in range(1,6):
    for j in range(1,6):
        print(i, j)
    