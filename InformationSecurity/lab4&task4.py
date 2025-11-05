## Caesar Cipher Implementation

# def caeser_encrypt(text, key):
#     result = ""
#     for ch in text:
#         if ch.isupper():
#             result += chr((ord(ch) - 65 + key) % 26 + 65)
#         else:
#             result += ch
#     return result
# def caeser_decrypt(cipher, key):
#     return caeser_encrypt(cipher, -key)
# encrypted = caeser_encrypt("HELLO", 3)
# print(encrypted)
# print("Decrypted:", caeser_decrypt(encrypted, 3))

## Monoalphabetic Substitution Cipher Implementation

# import string, random
# alphabet = list(string.ascii_uppercase)
# print("Alphabet:", alphabet)
# key = alphabet.copy()
# random.shuffle(key)
# print("Key:", key)
# encrypt_map = {}
# decrypt_map = {}
# for i in range(len(alphabet)):
#     encrypt_map[alphabet[i]] = key[i]
#     decrypt_map[key[i]] = alphabet[i]

# message = "HELLO"

# cipher = ""
# for letter in message:
#     if letter.isalpha():
#         cipher += encrypt_map[letter]
#     else:
#         cipher += letter

# plain = ""
# for letter in cipher:
#     if letter.isalpha():
#         plain += decrypt_map[letter]
#     else:
#         plain += letter

# print("Message:", message)
# print("Cipher:", cipher)
# print("Decrypted:", plain)

## Vigenère Cipher Implementation

# def vigenere_encrypt(text, key):
#     cipher = ""
#     key = key.upper()

#     for i, ch in enumerate(text.upper()):
#         if ch.isalpha():
#             shift = ord(key[i % len(key)]) - 65
#             cipher += chr((ord(ch) - 65 + shift) % 26 + 65)
#         else:
#             cipher += ch

#     return cipher

# def vigenere_decrypt(cipher, key):
#     text = ""
#     key = key.upper()

#     for i, ch in enumerate(cipher.upper()):
#         if ch.isalpha():
#             shift = ord(key[i % len(key)]) - 65
#             text += chr((ord(ch) - 65 - shift) % 26 + 65)
#         else:
#             text += ch

#     return text

# message = "HELLO VIGENERE"
# key = "KEY"

# encrypted = vigenere_encrypt(message, key)
# print("Encrypted:", encrypted)
# decrypted = vigenere_decrypt(encrypted, key)
# print("Decrypted:", decrypted)


# Task 1: Shift Cipher Encryption
#     • Write a Python program that:
#     • Takes a message (plaintext) from the user.
#     • Asks for a numeric key (e.g., 5).
#     • Encrypts the message by shifting each alphabet letter forward by that key.
#     • Decrypts the resulting ciphertext back to the original message.
#     • Ignore non-alphabetic characters during shifting (e.g., spaces or punctuation remain unchanged).
#     • Test your program by encrypting the following:
# “ YourName is doing Security Lab ” with a key = 5.



# def caesar_encrypt(text, key):
#     result = ""
#     for ch in text.upper():
#         if ch.isalpha():
#             result += chr((ord(ch) - 65 + key) % 26 + 65)
#         else:
#             result += ch
#     return result

# def caesar_decrypt(cipher, key):
#     return caesar_encrypt(cipher, -key)

# plaintext = input("Enter message: ")
# key = int(input("Enter numeric key: "))

# encrypted = caesar_encrypt(plaintext, key)
# decrypted = caesar_decrypt(encrypted, key)

# print("Encrypted:", encrypted)
# print("Decrypted:", decrypted)


# Task 2: Random Letter Substitution Cipher
#     • Generate a random mapping between each alphabet letter (A–Z) and its substitute letter.
#     • Use this mapping to:
#     • Encrypt a user-entered message.
#     • Decrypt it back to the original message using the same key map.
#     • The mapping should change each time the program runs (demonstrate randomness).
#     • Test your program by encrypting and decrypting your full name “your full name”.



# import string
# import random
# alphabet = list(string.ascii_uppercase)
# key = alphabet.copy()
# random.shuffle(key)
# encrypt_map = {}
# decrypt_map = {}
# for i in range(len(alphabet)):
#     encrypt_map[alphabet[i]] = key[i]
#     decrypt_map[key[i]] = alphabet[i]
# message = input("Enter your full name: ").upper()
# cipher = ""
# for letter in message:
#     if letter.isalpha():
#         cipher += encrypt_map[letter]
#     else:
#         cipher += letter
# plain = ""
# for letter in cipher:
#     if letter.isalpha():
#         plain += decrypt_map[letter]
#     else:
#         plain += letter
# print("Cipher:", cipher)
# print("Decrypted:", plain)

