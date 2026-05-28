import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans

# dataset
data = {
    "Age": [20,22,25,27,40,42,45,47],
    "Salary": [20000,22000,25000,27000,
               80000,82000,85000,87000]
}

df = pd.DataFrame(data)

model = KMeans(n_clusters = 4)

# train
model.fit(df)

# cluster label
label = model.labels_

print(label)

# visualization
plt.scatter(
    df["Age"],
    df["Salary"],
    c=label
)

plt.xlabel("Age")
plt.ylabel("Salary")

plt.show()