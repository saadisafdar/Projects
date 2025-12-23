# import matplotlib.pyplot as plt
# x = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
# y = [4000, 7000, 8000, 7500, 9000, 10000, 9500]
# plt.plot(x, y, color='green', linestyle='--', linewidth=2, marker='o', markersize=8)
# plt.title('Daily Steps Count')
# plt.xlabel('Days')
# plt.ylabel('Steps')
# plt.grid(True)
# plt.show()



# import matplotlib.pyplot as plt
# x = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
# y = [50, 65, 75, 100, 80]
# plt.bar(x, y, color='skyblue', edgecolor='black', linewidth=1.5, width=0.5)
# plt.title("Items sold per day")
# plt.xlabel("Days")
# plt.ylabel("Number of items")
# plt.grid(False)
# plt.show()



# import matplotlib.pyplot as plt
# scores = [45, 56, 78, 45, 60, 80, 90, 100, 70, 66, 77, 88, 95, 45, 60]
# plt.hist(scores, bins=5, color='orange', edgecolor='black', alpha=0.7, linewidth=1.5)
# plt.title('Score Distribution')
# plt.xlabel('Score Ranges')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()




# import matplotlib.pyplot as plt
# x = [1, 2, 3, 4, 5, 6, 7, 8]
# y = [30, 55, 60, 65, 70, 40, 80, 85]
# plt.scatter(x, y, color='purple', s=100, alpha=0.6, marker='^')
# plt.title("Study Hours vs. Test Score")
# plt.xlabel("Hours of Study")
# plt.ylabel("Score")
# plt.grid(True)
# plt.show()




# import matplotlib.pyplot as plt
# companies = ['Company A', 'Company B', 'Company C', 'Company D', 'Company E']
# market_share = [35, 25, 20, 15, 5]
# plt.pie(market_share, labels=companies, autopct='%1.1f%%', startangle=140, colors=['blue', 'orange', 'green', 'red', 'purple'])
# plt.title('Market Share of Tech Companies')
# plt.show()



# import matplotlib.pyplot as plt
# import pandas as pd
# data = {'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'], 
#         'Sales': [150, 200, 250, 300, 350]}
# df = pd.DataFrame(data)
# df.plot(x='Month', y='Sales', kind='area', color='yellow', alpha=0.5)
# plt.title('Cumulative Sales Over Time')
# plt.xlabel('Month')
# plt.ylabel('Sales')
# plt.grid(False)
# plt.show()




# import matplotlib.pyplot as plt  
# from mpl_toolkits.mplot3d import Axes3D # This is for 3D plotting in Matplotlib
# import pandas as pd 

# # Step 1: Create some sample data
# data = {
#     'Age': [23, 25, 30, 35, 40], 
#     'Height': [150, 160, 170, 180, 190],
#     'Weight': [50, 60, 70, 80, 90]
# }
# # Step 2: Convert the dictionary to a Pandas DataFrame
# df = pd.DataFrame(data) 
# # Step 3: Create a figure for the plot
# fig = plt.figure() # This creates a new figure window for the plot
# # Step 4: Add a 3D subplot to the figure
# # The '111' means that we are using 1 row, 1 column, and the first subplot (which is the only one here)
# ax = fig.add_subplot(111, projection='3d') # The 'projection="3d"' makes it a 3D plot
# # Step 5: Plot the data using a 3D scatter plot
# # The 'scatter' method is used for plotting individual data points in 3D space.
# ax.scatter(df['Age'], df['Height'], df['Weight'], color='red') 
# # Step 6: Customize the axes with labels
# ax.set_xlabel('Age')
# ax.set_ylabel('Height')
# ax.set_zlabel('Weight') 
# # Step 7: Add a title to the plot
# plt.title('3D Scatter Plot')
# # Step 8: Display the plot
# plt.show() 



# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt

# # Create DataFrame from dataset
# data = {'Subject': ['Math', 'English', 'Science', 'History', 'Computer'],
#     'Average_Score': [82, 75, 89, 70, 95]}
# df = pd.DataFrame(data)
# sns.barplot(x='Subject', y='Average_Score', data=df)
# plt.title('Average Scores by Subject')
# plt.xlabel('Subjects')
# plt.ylabel('Scores')
# plt.show()




# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt

# # Create DataFrame from dataset
# sales_data = {'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
#     'Revenue': [5000, 7000, 6500, 8500, 9000]}
# df = pd.DataFrame(sales_data)
# # Create line plot
# sns.lineplot(x='Month', y='Revenue', data=df, marker='o', color='green')
# plt.title('Monthly Revenue Trend')
# plt.xlabel('Month')
# plt.ylabel('Revenue ($)')
# plt.grid(True)
# plt.show()





# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt

# # Create DataFrame from dataset
# data = {'Scores': [45, 56, 67, 70, 90, 85, 88, 77, 50, 95]}
# df = pd.DataFrame(data)
# # Create histogram
# sns.histplot(data=df, x='Scores', bins=5, color='purple')
# plt.title('Distribution of coding Test Scores')
# plt.xlabel('Scores')
# plt.ylabel('Frequency')
# plt.show()





# Lab tasks
# 1. Line Plot(Matplotlib)
# Scenario:
# You are tracking your water intake for a week to improve your health. You record
# how many glasses of water you drink each day and want to visualize your daily
# progress.
# x = [&#39;Mon&#39;, &#39;Tue&#39;, &#39;Wed&#39;, &#39;Thu&#39;, &#39;Fri&#39;, &#39;Sat&#39;, &#39;Sun&#39;]
# y = [5, 6, 8, 7, 9, 10, 8]
# 2. Bar Chart
# Scenario:
# You run a small café and want to compare how many coffees were sold each day
# during the week to find your busiest day.
# x = [&#39;Mon&#39;, &#39;Tue&#39;, &#39;Wed&#39;, &#39;Thu&#39;, &#39;Fri&#39;, &#39;Sat&#39;, &#39;Sun&#39;]
# y = [45, 60, 50, 70, 85, 95, 90]
# 3. Histogram
# Scenario:
# You conducted a small survey among your classmates to record how many hours they
# sleep at night. You want to see how the sleeping hours are distributed among the
# students.
# sleep_hours = [5, 6, 7, 8, 5, 6, 7, 7, 8, 9, 5, 6, 8, 9, 7]
# 4. Scatter Plot
# Scenario:
# You are studying how much time students spend on social media versus their exam
# scores to see if more screen time affects their grades.
# x = [1, 2, 3, 4, 5, 6, 7, 8] # Hours spent on social media
# y = [95, 90, 85, 80, 75, 70, 60, 55] # Exam scores
# 5. Pie Chart
# Scenario:
# You analyzed your monthly expenses and want to visualize how your money is
# divided among categories like Food, Rent, Transport, Shopping, and Savings.
# categories = [&#39;Food&#39;, &#39;Rent&#39;, &#39;Transport&#39;, &#39;Shopping&#39;, &#39;Savings&#39;]
# expenses = [25, 35, 15, 10, 15]

# 6. Area Plot

# Scenario:
# You want to visualize how your savings have grown month by month throughout the
# year to see your total progress visually.
# &#39;Month&#39;: [&#39;Jan&#39;, &#39;Feb&#39;, &#39;Mar&#39;, &#39;Apr&#39;, &#39;May&#39;, &#39;Jun&#39;],
# &#39;Savings&#39;: [2000, 2500, 2700, 3000, 3500, 4000]
# 7. 3D Plot
# Scenario:
# You are analyzing the relationship between students’ age, study hours, and exam
# scores to understand how these three factors influence each other.
# &#39;Age&#39;: [18, 19, 20, 21, 22],
# &#39;Study_Hours&#39;: [2, 3, 4, 5, 6],
# &#39;Exam_Score&#39;: [60, 70, 75, 85, 90]

# 1. Bar Plot(seaborn)
# Scenario:
# You are a school teacher who wants to compare the average marks of students across
# different subjects to identify which subject needs more attention.
# Subject&#39;: [&#39;Math&#39;, &#39;English&#39;, &#39;Science&#39;, &#39;History&#39;, &#39;Computer&#39;]
# &#39;Average_Score&#39;: [82, 75, 89, 70, 95]
# 2. Line Plot
# Scenario:
# You are managing a delivery company and want to track how your monthly
# deliveries increased or decreased over the first half of the year.
# Month&#39;: [&#39;Jan&#39;, &#39;Feb&#39;, &#39;Mar&#39;, &#39;Apr&#39;, &#39;May&#39;, &#39;Jun&#39;],
# &#39;Deliveries&#39;: [120, 140, 135, 160, 180, 200]
# 3. Histogram
# Scenario:
# You are a fitness trainer and want to see how your clients’ weights are distributed to
# design personalized workout plans.
# data = {&#39;Weight&#39;: [55, 60, 62, 65, 70, 72, 75, 80, 85, 90, 95, 100]}


# import matplotlib.pyplot as plt
# x = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
# y = [5, 6, 8, 7, 9, 10, 8]
# plt.plot(x, y, color='blue', linestyle='-', linewidth=2, marker='s', markersize=8)
# plt.title('Daily Water Intake Over a Week')
# plt.xlabel('Days')
# plt.ylabel('Glasses of Water')
# plt.grid(True)
# plt.show()



# import matplotlib.pyplot as plt
# x = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
# y = [45, 60, 50, 70, 85, 95, 90]
# plt.bar(x, y, color='orange', edgecolor='black', linewidth=1.5, width=0.5)
# plt.title("Coffees Sold per Day")
# plt.xlabel("Days")
# plt.ylabel("Number of Coffees")
# plt.grid(False)
# plt.show()


# import matplotlib.pyplot as plt
# sleep_hours = [5, 6, 7, 8, 5, 6, 7, 7, 8, 9, 5, 6, 8, 9, 7]
# plt.hist(sleep_hours, bins=5, color='cyan', edgecolor='black', alpha=0.7, linewidth=1.5)
# plt.title('Distribution of Sleep Hours Among Students')
# plt.xlabel('Hours of Sleep')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()



# import matplotlib.pyplot as plt
# x = [1, 2, 3, 4, 5, 6, 7, 8] # Hours spent on social media
# y = [95, 90, 85, 80, 75, 70, 60, 55] # Exam scores
# plt.scatter(x, y, color='magenta', s=100, alpha=0.6, marker='o')
# plt.title("Social Media Hours vs. Exam Scores")
# plt.xlabel("Hours on Social Media")
# plt.ylabel("Exam Scores")
# plt.grid(True)
# plt.show()


# import matplotlib.pyplot as plt
# categories = ['Food', 'Rent', 'Transport', 'Shopping', 'Savings']
# expenses = [25, 35, 15, 10, 15]
# plt.pie(expenses, labels=categories, autopct='%1.1f%%', startangle=140, colors=['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'violet'])
# plt.title('Monthly Expenses Distribution')
# plt.show()



# import matplotlib.pyplot as plt
# import pandas as pd
# data = {'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], 
#         'Savings': [2000, 2500, 2700, 3000, 3500, 4000]}
# df = pd.DataFrame(data)
# df.plot(x='Month', y='Savings', kind='area', color='lightblue', alpha=0.5)
# plt.title('Cumulative Savings Over Months')
# plt.xlabel('Month')
# plt.ylabel('Savings ($)')
# plt.grid(False)
# plt.show()




# import matplotlib.pyplot as plt  
# from mpl_toolkits.mplot3d import Axes3D
# import pandas as pd
# data = {
#     'Age': [18, 19, 20, 21, 22], 
#     'Study_Hours': [2, 3, 4, 5, 6],
#     'Exam_Score': [60, 70, 75, 85, 90]
# }
# df = pd.DataFrame(data) 
# fig = plt.figure() 
# ax = fig.add_subplot(111, projection='3d') 
# ax.scatter(df['Age'], df['Study_Hours'], df['Exam_Score'], color='blue') 
# ax.set_xlabel('Age')
# ax.set_ylabel('Study Hours')
# ax.set_zlabel('Exam Score') 
# plt.title('3D Scatter Plot of Age, Study Hours, and Exam Score')
# plt.show()





# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# data = {'Subject': ['Math', 'English', 'Science', 'History', 'Computer'],
#     'Average_Score': [82, 75, 89, 70, 95]}
# df = pd.DataFrame(data)
# sns.barplot(x='Subject', y='Average_Score', data=df, palette='pastel')
# plt.title('Average Scores by Subject')
# plt.xlabel('Subjects')
# plt.ylabel('Scores')
# plt.show()




# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# sales_data = {'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
#     'Deliveries': [120, 140, 135, 160, 180, 200]}
# df = pd.DataFrame(sales_data)
# sns.lineplot(x='Month', y='Deliveries', data=df, marker='o', color='red')
# plt.title('Monthly Deliveries Trend')
# plt.xlabel('Month')
# plt.ylabel('Number of Deliveries')
# plt.grid(True)
# plt.show()


# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# data = {'Weight': [55, 60, 62, 65, 70, 72, 75, 80, 85, 90, 95, 100]}
# df = pd.DataFrame(data)
# sns.histplot(data=df, x='Weight', bins=5, color='teal')
# plt.title('Distribution of Client Weights')
# plt.xlabel('Weight (kg)')
# plt.ylabel('Frequency')
# plt.show()







# import matplotlib.pyplot as plt
# import numpy as np

# labels = ['G1', 'G2', 'G3', 'G4', 'G5']
# men_means = [20, 34, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]

# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, men_means, width, label='Men')
# rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(x, labels)
# ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

# fig.tight_layout()

# plt.show()

