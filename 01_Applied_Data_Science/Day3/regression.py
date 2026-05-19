import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
input_data = pd.read_csv("/home/shubham/ai-course/Day2/data.csv") 

# Prepare the data
input_data["Age"] = input_data["Age"].fillna(input_data["Age"].mean())
input_data["Salary"] = input_data["Salary"].fillna(input_data["Salary"].mean())
print(input_data["Age"])

# Define feature and target
x = input_data[["Age"]] #input
y = input_data["Salary"] #output

# TRAIN-TEST split
## Check if model GENERALIZES
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size = 0.2, random_state = 42
)

# Train Model using Linear regression
model = LinearRegression()
model.fit(x_train, y_train)

# Make Predication 
y_pred = model.predict(x_test)
print(y_pred)

# Evalution Model
mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)

# Visualization
plt.scatter(x, y)
plt.plot(x, model.predict(x), color="red")
plt.show()