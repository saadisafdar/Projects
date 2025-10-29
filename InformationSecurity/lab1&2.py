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