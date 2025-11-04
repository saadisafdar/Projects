import numpy as np

print("\n== Create simple array ==")
arr = np.array([1, 2, 3])
print(arr)

print("\n== arange ==")
arr_arr = np.arange(1, 10, 2)
print(arr_arr)

print("\n== arange with float dtype ==")
arr_dt = np.arange(1, 10, 2, dtype=float)
print(arr_dt)

print("\n== linspace ==")
ary = np.linspace(0, 10, 5)
print(ary)

print("\n== zeros and ones ==")
arr_zeros = np.zeros((2, 3))
arr_ones = np.ones((3, 2))
print(arr_zeros)
print(arr_ones)

print("\n== full ==")
arr_full = np.full((2, 2), 7)
print(arr_full)

print("\n== random ==")
arr_random = np.random.rand(3, 3)
print(arr_random)

print("\n== array properties ==")
ar = np.array([[1, 2, 3], [4, 5, 6]])
print("Data Type:", ar.dtype)
print("Shape:", ar.shape)
print("Size:", ar.size)
print("Dimensions:", ar.ndim)

print("\n== arithmetic operations ==")
arr_ath = np.array([1, 2, 3])
print("Addition:", arr_ath + 5)
print("Multiplication:", arr_ath * 2)

print("\n== element-wise addition ==")
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print("Element-wise Addition:", arr1 + arr2)

print("\n== indexing ==")
arr_index = np.array([10, 20, 30, 40, 50])
print(arr_index[2])

print("\n== slicing ==")
arr_slice = np.array([1, 2, 3, 4, 5, 6])
print(arr_slice[1:4])

print("\n== math functions ==")
arr_sl = np.array([1, 4, 9, 16])
print("Square Root:", np.sqrt(arr_sl))
print("Logarithm:", np.log(arr_sl))

print("\n== statistical functions ==")
arr_mm = np.array([10, 20, 30, 40, 50])
print("Mean:", np.mean(arr_mm))
print("Median:", np.median(arr_mm))
print("Standard Deviation:", np.std(arr_mm))

print("\n== reshape ==")
arr_reshape = np.array([1, 2, 3, 4, 5, 6])
reshaped = arr_reshape.reshape((2, 3))
print(reshaped)

print("\n== concatenate ==")
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
print("Concatenated Array:", np.concatenate((array1, array2)))


import pandas as pd

print("\n== DataFrame from list ==")
data = [['Ali', 22], ['Ayesha', 21], ['Usman', 23]]
df1 = pd.DataFrame(data, columns=['Name', 'Age'])
print(df1)

print("\n== DataFrame from dictionary ==")
data_dict = {
    'Name': ['Ali', 'Ayesha', 'Usman'],
    'Age': [22, 21, 23],
    'Grade': ['A', 'B', 'A']
}
df2 = pd.DataFrame(data_dict)
print(df2)

print("\n== Read data from CSV ==")
df = pd.read_csv('students.csv')
print(df.head())

print("\n== Display specific number of rows ==")
print(df.head(10))
print(df.tail(3))
print(df.sample(2))

print("\n== Accessing data ==")
print(df['Name'])
print(df[['Name', 'Age']])

print("\n== Filtering and sorting ==")
print(df[df['Age'] > 21])
print(df.sort_values(by='Age'))
print(df.sort_values(by='Age', ascending=False))

print("\n== Grouping and aggregation ==")
print(df.groupby('Grade').size())
print(df.groupby('Grade')['Age'].mean())
