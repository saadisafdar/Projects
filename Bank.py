class Bank:
    accounts_created = 0  # Class variable

    def __init__(self, name):
        self.name = name
        Bank.accounts_created += 1

    @classmethod
    def how_many_accounts(cls):
        print(f"👥 Total accounts created: {cls.accounts_created}")

b1 = Bank("Bank A")
b2 = Bank("Bank B") 
b3 = Bank("Bank C")
Bank.how_many_accounts()  # Output: 👥 Total accounts created: 2
