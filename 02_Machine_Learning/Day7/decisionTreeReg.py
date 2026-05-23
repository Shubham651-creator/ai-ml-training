import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv("../Day6/Day6_data.csv")
df = pd.DataFrame(data)
print(df)

# Prepare the data
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Experience"] = df["Experience"].fillna(df["Experience"].mean())
df["Education_level"] = df["Education_level"].fillna(df["Education_level"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

# Define feture and target variables
x = df[["Age", "Experience","Education_level"]]
y = df["Salary"]

# Train-Test split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size = 0.2, random_state = 42
)

# Train model using Decision Tree Regression
model = DecisionTreeRegressor(max_depth=20)
model.fit(x_train, y_train)

# make Prediction
y_pred = model.predict(x_test)
print(y_pred)

# Evalution Model
mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)

# visulization for single feature by scatter plotting
plt.scatter(df["Age"],y, label="Actual data")
plt.scatter(df["Age"], model.predict(x), color="red", label="Prediction")

plt.xlabel("Age")
plt.ylabel("Salary")

plt.legend()
plt.show()