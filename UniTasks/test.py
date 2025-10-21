print("Python setup OK")
print("All systems go!")# calculator.py

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "Error: Division by zero"
    return a / b

print("Simple Calculator")
print("Operations: +  -  *  /")

while True:
    a = float(input("Enter first number: "))
    op = input("Enter operation (+ - * / or q to quit): ")
    if op.lower() == 'q':
        print("Goodbye.")
        break
    b = float(input("Enter second number: "))

    if op == '+':
        print("Result:", add(a, b))
    elif op == '-':
        print("Result:", subtract(a, b))
    elif op == '*':
        print("Result:", multiply(a, b))
    elif op == '/':
        print("Result:", divide(a, b))
    else:
        print("Invalid operation.")
