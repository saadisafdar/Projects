import bcrypt
import random
import string

class PasswordManager:
    """
    A class to manage password generation, validation, and hashing.
    """

    @staticmethod
    def generate_random_password(length=12):
        """
        Generate a secure random password.
        """
        characters = string.ascii_letters + string.digits + string.punctuation
        password = ''.join(random.choice(characters) for _ in range(length))
        return password

    @staticmethod
    def hash_password(password):
        """
        Hash a password using bcrypt.
        """
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        return hashed

    @staticmethod
    def is_password_strong(password):
        """
        Validate the strength of a password.
        """
        if len(password) < 8:
            return False
        if not any(char.isdigit() for char in password):
            return False
        if not any(char.isupper() for char in password):
            return False
        if not any(char.islower() for char in password):
            return False
        if not any(char in string.punctuation for char in password):
            return False
        return True

# Example usage
if __name__ == '__main__':
    pm = PasswordManager()
    new_password = pm.generate_random_password()  
    print(f'Generated Password: {new_password}')
    print(f'Is the password strong? {pm.is_password_strong(new_password)}')
    hashed_password = pm.hash_password(new_password)
    print(f'Hashed Password: {hashed_password.decode()}')
