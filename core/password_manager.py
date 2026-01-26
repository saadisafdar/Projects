# Password Manager

"""
This script handles secure password storage and management.
"""

import hashlib
import os
import json

class PasswordManager:
    def __init__(self, storage_file='passwords.json'):
        self.storage_file = storage_file
        self.load_passwords()

    def load_passwords(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r') as f:
                self.passwords = json.load(f)
        else:
            self.passwords = {}

    def save_passwords(self):
        with open(self.storage_file, 'w') as f:
            json.dump(self.passwords, f)

    def add_password(self, service, password):
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        self.passwords[service] = hashed_password
        self.save_passwords()

    def get_password(self, service):
        return self.passwords.get(service, None)

# Example usage
if __name__ == '__main__':
    pm = PasswordManager()
    pm.add_password('email', 'securepassword123')
    print(pm.get_password('email'))