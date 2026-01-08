# from Crypto.Cipher import AES
# from Crypto.Random import get_random_bytes
# import base64

# key = get_random_bytes(16)
# iv = get_random_bytes(16)

# plaintext = "Hello, World! This is a test message."

# def pad(text):
#     while len(text) % 16 != 0:
#         text += ' '
#     return text

# cipher = AES.new(key, AES.MODE_CBC, iv)
# padded_text = pad(plaintext).encode('utf-8')
# cipher_text = cipher.encrypt(padded_text)

# cipher_text_base64 = base64.b64encode(cipher_text).decode('utf-8')
# print("Encrypted Text (Base64):", cipher_text_base64)

# cipher_decrypt = AES.new(key, AES.MODE_CBC, iv)
# decrypted_text = cipher_decrypt.decrypt(cipher_text).decode('utf-8').rstrip()
# print("Decrypted Text:", decrypted_text)





from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

plaintext = b"My Name is Saadi"
key = b"ThisIsA16ByteKey"
iv = b"\x00" * 16

cipher = AES.new(key, AES.MODE_CBC, iv=iv)
ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

print("Ciphertext (hex):", ciphertext.hex())

cipher_decrypt = AES.new(key, AES.MODE_CBC, iv=iv)
decrypted_text = cipher_decrypt.decrypt(ciphertext)
print("Decrypted Text:", decrypted_text.rstrip(b'\x00').decode('utf-8'))



