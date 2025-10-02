class BankAccount:
    def __init__(self, name, balance=0):
        self.name = name
        self.balance = balance

    def show_balance(self):
        print(f"{self.name}'s balance is {self.balance}")

    def deposit(self, amount):
        self.balance += amount
        print(f"💰 Deposited {amount}. New balance: {self.balance}")

    def withdraw(self, amount):
        if amount > self.balance:
            print("❌ Insufficient balance!")
        else:
            self.balance -= amount
            print(f"💸 Withdrawn {amount}. New balance: {self.balance}")

class SavingsAccount(BankAccount):
    def __init__(self, name, balance=0, interest_rate=0.02):
        super().__init__(name, balance)  # Inherit parent's __init__
        self.interest_rate = interest_rate

    def apply_interest(self):
        interest = self.balance * self.interest_rate
        self.balance += interest
        print(f"💹 Interest of {interest} added. New balance: {self.balance}")

sa = SavingsAccount("Saadi", 1000)
sa.show_balance()
sa.apply_interest()
sa.withdraw(500)
sa.show_balance()
sa.deposit(200)
sa.show_balance()
sa.withdraw(800)  # Should show insufficient balance
sa.show_balance()
sa.apply_interest()  # Apply interest again
sa.show_balance()                   

