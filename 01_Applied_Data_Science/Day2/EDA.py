import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load data
data = {
    "Name": ["Sonali", "Priyankya", "Karishma", "Suraj"],
    "Age":[18,25,None, 26],
    "Salary": [25000, 10000, 36460, 45690]
}
result = pd.DataFrame(data) 

# Step 2: Inspect data
print(result.head(2))
print(result.columns)
print(result.info()) # Data types + Nulls
print(result.describe()) # Statistics

# Step 3: Clean the data
## Handle missing values
print(result.fillna(value = 18)) #Fill all NaN with give value
print(pd.isna(result)) # Show Table with boolean values
print(result.dropna()) # Drop the row that has NaN

#Step 4: Visualization
## Graph
plt.plot( result["Salary"], result["Age"])
plt.xlabel("Salary")
plt.ylabel("Age")
plt.title("Salary Vs. Age")
plt.show()

## Bar
plt.bar( result["Name"], result["Age"])
plt.show()
