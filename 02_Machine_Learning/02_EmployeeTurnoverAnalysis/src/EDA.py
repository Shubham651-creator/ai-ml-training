import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df = pd.DataFrame(data)

# # Inspect the data
# print(df.head())
# print(df.info()) # columns, types
# print(df.describe()) # statistics 

# Handling missing the data
df.dropna() 

# Visualization
print("1. How many employee stay & left?")
## -> Stay = 237 & left = 1237 employee

attrition_count = df["Attrition"].value_counts()
plt.bar(
    attrition_count.index,
    attrition_count.values
)

plt.xlabel("Attrition")
plt.ylabel("Employee Count")
# plt.show()

## 2. do employee with overtime leaves more?
print("\n### 2. Attrition vs Overtime ###")
print(
    pd.crosstab(
        
        df["OverTime"],
        df["Attrition"],
        normalize="index"
    ) *100
)

## 3. What department loses most employee?
print("\n### 3. Attrition vs Department ###")
print(
    pd.crosstab(
     
        df["Department"],
        df["Attrition"],
        normalize="index"
    ) *100
)

## 4. Does job satification affects attrition?
print("\n### 4. Attrition vs JobSatisfaction ###")
print(
    pd.crosstab(
        df["JobSatisfaction"],
        df["Attrition"],
        normalize="index"
    ) *100
)

## 5. Does BusinessTravel affects attrition?
print("\n### 5. Attrition vs BusinessTravel ###")
print(
    pd.crosstab(
        df["BusinessTravel"],
        df["Attrition"],
        normalize="index"
    ) *100
)

## 6. Do lower-income employees leave more?
print("\n### 6. Attrition vs MonthlyIncome ###")
print(
    pd.crosstab(
        pd.qcut(df["MonthlyIncome"], 5),
        df["Attrition"],
        normalize="index"
    ) * 100
)

print("\nSplit data into groups:")
print(df.groupby("Attrition")["YearsAtCompany"].mean())

print("\nSelect ONLY numeric columns:")
numeric_df = df.select_dtypes(include="number")
print(numeric_df.corr())

print(df["EmployeeCount"].unique())
print(df["StandardHours"].unique())