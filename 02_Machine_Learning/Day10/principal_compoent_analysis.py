import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv("../Day6/Day6_data.csv")
df = pd.DataFrame(data)

# scaling important for PCA
scalar = StandardScaler()
scalar_data = scalar.fit_transform(df)

# PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scalar_data)

print(reduced_data)