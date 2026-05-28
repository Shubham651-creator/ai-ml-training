import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans

# # dataset
# data = {
#     "Age": [20,22,25,27,40,42,45,47],
#     "Salary": [20000,22000,25000,27000,
#                80000,82000,85000,87000]
# }

# Load the data
data = pd.read_csv("../Day6/Day6_data.csv")

df = pd.DataFrame(data)

wcss=[]

for k in range(1, 8):
    model = KMeans(n_clusters = k, random_state=42)

    # train
    model.fit(df)

    # cluster label
    wcss.append(model.inertia_) #this gives wcss values

# visualization of elbow graph
plt.plot(range(1,8), wcss)

plt.xlabel("Number of cluster(K)")
plt.ylabel("wcss")

plt.title("Elbow method")

plt.show()