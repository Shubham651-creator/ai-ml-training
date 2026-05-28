# Unsupervised Learning 
- No label exits, Model finds hidden pattern itself

# Clustering
> Grouping similar data points together.

## 1. K-mean clustering Algorithm
- k=3 (Create 3 cluster/groups)

### Limitation of Kmean
1. Must choose k manually
2. Sensitive to outliers
3. Assumes cicular clusters

## 2. Elbow Method
> How do we choose correct K?
> What is the best number of clusters?

### WCSS (Within Cluster Sum of Square)
- We plot graph x=k and y=wcss value, choose bend value as optimal k

---

# PCA (Principal Component Analysis)
> What happens when features become too many?
> (Curse of Dimensionality)

- PCA teaches that not all features are equally important

## Dimensionality Reduction
- keep important feature and remove unnessary dimensions.
- Compression of information
- PCA tries to preserve maximum information (variance) while reducing dimension
- more variance = more information

### PCA usecase
1. Image compression
2. Face recognition
3. Bioinformatic- gene compression

### PCA Limitation
1. loses interpretability
2. may lose information
3. assumes liner structure