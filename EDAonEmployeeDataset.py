#!/usr/bin/env python
# coding: utf-8

import pandas as pd

# In[34]:

#make sure to give proper path to the dataset
data = pd.read_csv('/kaggle/input/5000-employee-dataset-of-an-organization-dummy/Dummy_5000_Employee_Details_Dataset.csv')

# In[35]:

print(data.head())

# In[36]:


print(data.info())
print(data.describe(include='all'))


# In[37]:


# Analyze age distribution
import matplotlib.pyplot as plt
data['Age'].hist(bins=40)  # Adjust bin count as needed
plt.xlabel('Age')
plt.ylabel('Number of Employees')
plt.title('Distribution of Employee Age')
plt.show()


# In[38]:


# Analyze salary distribution
data['Salary'].hist(bins=50)
plt.xlabel('Salary')
plt.ylabel('Number of Employees')
plt.title('Distribution of Employee Salaries')
plt.show()


# In[39]:


# Analyze experience distribution
data['Year of Experience'].hist(bins=40)
plt.xlabel('Years of Experience')
plt.ylabel('Number of Employees')
plt.title('Distribution of Employee Experience')
plt.show()


# In[40]:


# Correlation analysis (find relationships between features)
correlation = data.corr()
print(correlation)


# In[41]:


# Group by department and analyze average salary
average_salary_by_department = data.groupby('Department')['Salary'].mean()
print(average_salary_by_department)


# In[42]:


# Group by marital status and analyze average dependents
average_dependents_by_marital_status = data.groupby('Marital Status')['Dependents'].mean()
print(average_dependents_by_marital_status)


# In[43]:


# 1. Central Tendency Measures

# Mean (Average) salary
average_salary = data['Salary'].mean()
print(f"Average Salary: INR {average_salary:.2f}")

# Median salary
median_salary = data['Salary'].median()
print(f"Median Salary: INR {median_salary:.2f}")

# Mode of job positions
most_common_position = data['Position'].mode()[0]
print(f"Most Common Job Position: {most_common_position}")


# In[44]:


# 2. Dispersion Measures

# Standard deviation of salary
salary_std = data['Salary'].std()
print(f"Standard Deviation of Salary: INR {salary_std:.2f}")

# Interquartile Range (IQR) for experience
q1 = data['Year of Experience'].quantile(0.25)
q3 = data['Year of Experience'].quantile(0.75)
iqr = q3 - q1
print(f"Interquartile Range (IQR) for Long Experience: {iqr:.2f} years")

q2 = data['Year of Experience'].quantile(0.40)
q4 = data['Year of Experience'].quantile(0.60)
iqr2 = q4 - q2
print(f"Interquartile Range (IQR) for Short Experience: {iqr2:.2f} years")


# In[58]:


# 3. Rates and Ratios

# Assuming data for number of employees leaving and average number of employees is available (replace with your data source)
number_leaving = 100  # Replace with actual value
average_employees = 4500  # Replace with actual value
turnover_rate = (number_leaving / average_employees) * 100
print(f"Employee Turnover Rate: {turnover_rate:.2f}%")

# Assuming data for number of employees and managers per department is available (replace with your data source)
department = "IT"  # Replace with specific department
department_data = data[data['Department'] == department]
employee_count = department_data.shape[0]
manager_count = len(department_data[department_data['Position'].str.contains('Manager')])  # Assuming 'Manager' in position title
employee_to_manager_ratio = employee_count / manager_count
print(f"Employee-to-Manager Ratio in {department}: {employee_to_manager_ratio:.2f}")


# In[69]:


# Correlation between salary and experience
correlation = data['Salary'].corr(data['Year of Experience'])
print(f"Correlation between Salary and Experience: {correlation:.2f}")

# 5. Hypothesis Testing (using Chi-Square Test example)
# Assuming data for department and gender is available (replace with your data source)
from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(data['Department'], data['Sex'])
chi2, pval, deg_of_freedom, expected_freq = chi2_contingency(contingency_table.values)

# Interpretation of p-value (adjust significance level as needed)
if pval < 0.05:
  print("There is a statistically significant relationship between department and gender distribution.")
else:
  print("There is no statistically significant relationship between department and gender distribution.")


# In[71]:


print(contingency_table)
print(expected_freq)
print(deg_of_freedom)
print(pval)
print(chi2)


# In[74]:


# Boxplot for salary distribution
plt.boxplot(data['Salary'])
plt.xlabel('Salary')
plt.ylabel('Amount')
plt.title('Distribution of Employee Salaries (Boxplot)')
plt.show()


# In[75]:


# Boxplot for experience distribution (optional, adjust labels as needed)
plt.boxplot([data['Year of Experience'], data[data['Position'].str.contains('Manager')]['Year of Experience']])  # Assuming 'Manager' in position title
plt.xlabel('Employee Category')
plt.ylabel('Years of Experience')
plt.title('Distribution of Employee Experience by Category (Boxplot)')
plt.xticks([1, 2], ['All Employees', 'Managers'])
plt.show()


# In[76]:


# Scatter plot for salary vs experience
plt.scatter(data['Year of Experience'], data['Salary'])
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs. Experience (Scatter Plot)')
plt.show()


# In[81]:


import seaborn as sns
correlation_matrix = data[['Year of Experience', 'In Company Years']].corr()  # Select relevant columns
print(correlation_matrix)
#if this value is less means no of experienced employees are fom the different companies who joined later on this company
# Create heatmap
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Heatmap (Experience and Salary)')
plt.show()


# In[86]:


experience_groups = pd.cut(data['Year of Experience'], bins=[0, 5, 10, 15, 20, 25, 30, 35, 40])
data_segmented = data.groupby(experience_groups)

# Analyze average salary for each experience group
average_salary_by_experience = data_segmented['Salary'].mean()
print(average_salary_by_experience)


# In[96]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features = ['Salary', 'Year of Experience']
data_scaled = StandardScaler().fit_transform(data[features])  # Scale features

# Choose the number of clusters (experiment with different values)
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(data_scaled)

# Assign cluster labels to data points
data['cluster'] = kmeans.labels_

# Analyze characteristics of each cluster (e.g., average salary, experience)
print(data.groupby('cluster').describe())

