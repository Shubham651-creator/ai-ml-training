import pandas as pd
import matplotlib.pyplot as plt

# load
data = {
    "Experience": [2,5,3,4]
}
load_values = pd.DataFrame(data)
print(load_values)

# Perforn EDA
print(load_values.head())
print(load_values.info())
print(load_values.describe())

# Handling missing values
mean_value = load_values.mean()
load_values.dropna()
print(mean_value)

# Visualization
## Graph
plt.plot(load_values["Experience"])
plt.xlabel("Experience")
plt.show()