import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv("Day6_data.csv")
df = pd.DataFrame(data)
print(df)

# Prepare the data
data["Age"] = data["Age"].fillna(data["Age"].mean())
data["Experience"] = data["Experience"].fillna(data["Experience"].mean())
data["Education_level"] = data["Education_level"].fillna(data["Education_level"].mean())
data["Salary"] = data["Salary"].fillna(data["Salary"].mean())

# Define feture and target variables
x = df[["Age", "Experience","Education_level"]]
y = df["Salary"]

# Train-Test split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size = 0.2, random_state = 42
)

# Train model using linear Regression
model = LinearRegression()
model.fit(x_train, y_train)

# make Prediction
y_pred = model.predict(x_test)
print(y_pred)

# Model output weight
print(model.coef_)

# Evalution Model
mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)

# visulization for single feature
plt.scatter(df["Age"],y)
plt.plot(x, model.predict(x), color="red")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()