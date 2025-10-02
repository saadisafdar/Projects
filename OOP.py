class Car:
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year
    
    def display(self):
        print(f"This is a {self.year} {self.brand} {self.model}.")


c1 = Car("Toyota", "Corolla", 2020)
c2 = Car("Honda", "Civic", 2021)
c3 = Car("Ford", "Mustang", 2022)
c4 = Car("Tesla", "Model 3", 2023)

c1.display()
c2.display()
c3.display()
c4.display()
