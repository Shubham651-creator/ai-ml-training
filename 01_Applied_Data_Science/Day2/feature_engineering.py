import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data
employee_data = pd.read_csv("data.csv") 

# Data inspection
print(employee_data.info())
print(employee_data.isnull().sum())

# Data cleaning
employee_data["Age"] = employee_data["Age"].fillna(employee_data["Age"].mean())
employee_data["Salary"] = employee_data["Salary"].fillna(employee_data["Salary"].mean())

# FEATURE ENGINEERING
## Create New feature
employee_data["Age_Salary"] = employee_data["Age"].apply(lambda x: "Young" if x < 30 else "Senior")

## Encode Categrical Data
employee_data = pd.get_dummies(employee_data, columns=["Department"])
print(employee_data)

## Feature Scaling
scalar = StandardScaler()
employee_data["Salary"] = scalar.fit_transform(employee_data[["Salary"]])
print(employee_data)