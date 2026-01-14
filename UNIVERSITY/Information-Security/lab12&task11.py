# def modInverse(e, phi):
#     for d in range(2, phi):
#         if (e * d) % phi == 1:
#             return d
#     return -1

# # RSA Key Generation
# def generateKeys():
#     p = 7919
#     q = 1009

#     n = p * q
#     phi = (p - 1) * (q - 1)
#     # Choose e, where 1 < e < phi(n) and gcd(e, phi(n)) == 1
#     e = 0
#     for e in range(2, phi):
#         if gcd(e, phi) == 1:
#             break
#     # Compute d such that e * d ≡ 1 (mod phi(n))
#     d = modInverse(e, phi)
#     return e, d, n

# # Function to calculate gcd
# def gcd(a, b):
#     while b != 0:
#         a, b = b, a % b
#     return a

# # Encrypt message using public key (e, n)
# def encrypt(m, e, n):
#     return pow(m, e, n)

# # Decrypt message using private key (d, n)
# def decrypt(c, d, n):
#     return pow(c, d, n)


# # Key Generation
# e, d, n = generateKeys()

# print(f"Public Key (e, n): ({e}, {n})")
# print(f"Private Key (d, n): ({d}, {n})")

# # Message
# M = 123
# print(f"Original Message: {M}")

# # Encrypt the message
# C = encrypt(M, e, n)
# print(f"Encrypted Message: {C}")

# # Decrypt the message
# decrypted = decrypt(C, d, n)
# print(f"Decrypted Message: {decrypted}")





# # Diffie-Hellman Key Exchange
# # Step 1: Agree on prime (p) and generator (g)
# p = 7
# g = 5

# # Step 2: Private keys
# xa = 3
# xb = 4

# # Step 3: Public keys
# ya = pow(g, xa, p)
# yb = pow(g, xb, p)

# print(f"Alice's public key: {ya}")
# print(f"Bob's public key: {yb}")

# # Step 4: Shared secret calculation
# k_alice = pow(yb, xa, p)
# k_bob = pow(ya, xb, p)

# print(f"Alice's calculated shared secret: {k_alice}")
# print(f"Bob's calculated shared secret: {k_bob}")







p = 101
g = 88

xa = 4
xb = 7

ya = pow(g, xa, p)
yb = pow(g, xb, p)

print(f"Alice's public key: {ya}")
print(f"Bob's public key: {yb}")

k_alice = pow(yb, xa, p)
k_bob = pow(ya, xb, p)

print(f"Alice's calculated shared secret: {k_alice}")
print(f"Bob's calculated shared secret: {k_bob}")





def modInverse(e, phi):
    for d in range(2, phi):
        if (e * d) % phi == 1:
            return d
    return -1

# RSA Key Generation
def generateKeys():
    p = 61
    q = 53

    n = p * q
    phi = (p - 1) * (q - 1)
    # Choose e, where 1 < e < phi(n) and gcd(e, phi(n)) == 1
    e = 0
    for e in range(2, phi):
        if gcd(e, phi) == 1:
            break
    # Compute d such that e * d ≡ 1 (mod phi(n))
    d = modInverse(e, phi)
    return e, d, n

# Function to calculate gcd
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

# Encrypt message using public key (e, n)
def encrypt(m, e, n):
    return pow(m, e, n)

# Decrypt message using private key (d, n)
def decrypt(c, d, n):
    return pow(c, d, n)


# Key Generation
e, d, n = generateKeys()

print(f"Public Key (e, n): ({e}, {n})")
print(f"Private Key (d, n): ({d}, {n})")

# Message
M = 123
print(f"Original Message: {M}")

# Encrypt the message
C = encrypt(M, e, n)
print(f"Encrypted Message: {C}")

# Decrypt the message
decrypted = decrypt(C, d, n)
print(f"Decrypted Message: {decrypted}")