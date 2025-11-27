# import pandas as pd

# # Create data
# data = {
#     'Name': ['Ali', 'Ayesha', 'Usman'],
#     'Age': [22, 21, 23],
#     'Grade': ['A', 'B', 'A'],
#     'Marks': [85, 76, 92]
# }

# # Create DataFrame
# df = pd.DataFrame(data)

# # Save as CSV
# df.to_csv('my_students.csv', index=False)
# print("CSV file created successfully!")


# import pandas as pd

# # Read the CSV file
# df = pd.read_csv('students.csv')

# # Display the data
# print("Complete DataFrame:")
# print(df)



# df = pd.read_csv('students.csv')

# print("Type:", type(df))        # <class 'pandas.core.frame.DataFrame'>
# print("Shape:", df.shape)       # (5, 5) - 5 rows, 5 columns
# print("Columns:", df.columns)   # Index(['Name', 'Age', 'Grade', 'Marks', 'City'])
# print("\nFirst 2 rows:")
# print(df.head(2))










# import pandas as pd

# # Read CSV
# df = pd.read_csv('Students.csv')

# # 1. View data
# print("First 3 rows:")
# print(df.head(3))

# print("\nLast 2 rows:")
# print(df.tail(2))

# # 2. Access specific columns
# print("\nJust names:")
# print(df['Name'])

# print("\nNames and marks:")
# print(df[['Name', 'Marks']])

# # 3. Filter data
# print("\nStudents with marks > 80:")
# good_students = df[df['Marks'] > 80]
# print(good_students)

# # 4. Add new column
# df['Status'] = ['Pass' if marks >= 50 else 'Fail' for marks in df['Marks']]
# print("\nWith status column:")
# print(df)

# # 5. Basic statistics
# print("\nStatistics:")
# print("Average marks:", df['Marks'].mean())
# print("Highest marks:", df['Marks'].max())
# print("Number of students:", len(df))

# # 6. Save modified data
# df.to_csv('updated_students.csv', index=False)
# print("\nNew CSV file saved!")






# import pandas as pd

# # Read CSV
# df = pd.read_csv('products.csv')

# # Calculate total value
# df['Total_Value'] = df['Price'] * df['Quantity']

# # Filter electronics
# electronics = df[df['Category'] == 'Electronics']

# # Save results
# df.to_csv('products_with_values.csv', index=False)
# electronics.to_csv('electronics_products.csv', index=False)

# print("Original data:")
# print(df)
# print("\nElectronics products:")
# print(electronics)





# import pandas as pd

# # Read CSV
# df = pd.read_csv('students_data.csv')

# # Calculate total and average marks
# df['Total_Marks'] = df['Math_Marks'] + df['Science_Marks'] + df['English_Marks']
# df['Average_Marks'] = df['Total_Marks'] / 3

# # Add status column
# df['Status'] = ['Pass' if avg >= 60 else 'Fail' for avg in df['Average_Marks']]

# # Find failed students
# failed_students = df[df['Status'] == 'Fail']

# # Save results
# df.to_csv('student_report.csv', index=False)
# failed_students.to_csv('failed_students.csv', index=False)

# print("Complete Report:")
# print(df)
# print("\nFailed Students:")
# print(failed_students)






# import pandas as pd

# # Read CSV
# df = pd.read_csv('sales.csv')

# # Calculate revenue
# df['Total_Revenue'] = df['Units_Sold'] * df['Price_Per_Unit']

# # Total revenue by category
# category_revenue = df.groupby('Category')['Total_Revenue'].sum()

# # Best-selling product
# best_seller = df[df['Units_Sold'] == df['Units_Sold'].max()]

# # Save results
# category_revenue.to_csv('category_sales.csv')
# df.to_csv('sales_with_revenue.csv', index=False)

# print("Sales Data with Revenue:")
# print(df)
# print("\nRevenue by Category:")
# print(category_revenue)
# print("\nBest Selling Product:")
# print(best_seller)






# import pandas as pd

# # Read CSV
# df = pd.read_csv('employees.csv')

# # Average salary by department
# avg_salary_dept = df.groupby('Department')['Salary'].mean()

# # Experienced employees
# experienced = df[df['Experience'] > 3]

# # Employees per city
# city_count = df['City'].value_counts()

# # Add bonus column
# df['Bonus'] = [salary * 0.10 if exp >= 3 else salary * 0.05 
#                for salary, exp in zip(df['Salary'], df['Experience'])]

# # Save results
# df.to_csv('employees_with_bonus.csv', index=False)

# print("Employee Data with Bonus:")
# print(df)
# print("\nAverage Salary by Department:")
# print(avg_salary_dept)
# print("\nExperienced Employees (>3 years):")
# print(experienced)







# import pandas as pd

# # Read CSV
# df = pd.read_csv('patients.csv')

# # Patients with fever
# fever_patients = df[df['Temperature'] > 99.5]

# # Patients with high BP
# high_bp_patients = df[(df['BP_Systolic'] > 130) | (df['BP_Diastolic'] > 85)]

# # Average age
# average_age = df['Age'].mean()

# # Blood group count
# blood_group_count = df['Blood_Group'].value_counts()

# # Critical patients (fever OR high BP)
# critical_patients = df[(df['Temperature'] > 99.5) | 
#                        (df['BP_Systolic'] > 130) | 
#                        (df['BP_Diastolic'] > 85)]

# # Save results
# critical_patients.to_csv('critical_patients.csv', index=False)

# print("All Patients:")
# print(df)
# print("\nCritical Patients:")
# print(critical_patients)
# print(f"\nAverage Age: {average_age:.1f} years")
# print("\nBlood Group Distribution:")
# print(blood_group_count)










# import pandas as pd

# # Read CSV
# df = pd.read_csv('books.csv')

# # Calculate inventory value
# df['Inventory_Value'] = df['Price'] * df['Stock']

# # Education books
# education_books = df[df['Genre'] == 'Education']

# # Low stock books
# low_stock = df[df['Stock'] < 20]

# # Most expensive book
# most_expensive = df[df['Price'] == df['Price'].max()]

# # Save results
# education_books.to_csv('education_books.csv', index=False)
# low_stock.to_csv('low_stock_books.csv', index=False)

# print("All Books with Inventory Value:")
# print(df)
# print("\nEducation Books:")
# print(education_books)
# print("\nLow Stock Books (< 20):")
# print(low_stock)
# print("\nMost Expensive Book:")
# print(most_expensive)