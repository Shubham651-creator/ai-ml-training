import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# data
X = np.array([1,2,3,4,5,6,7,8,9]).reshape(-1,1)
y = np.array([1,4,9,16,25,36,49,64,91])

# polynomial transformation
poly = PolynomialFeatures(degree=50)

X_poly = poly.fit_transform(X)

# train model
model = LinearRegression()
model.fit(X_poly, y)

# prediction
y_pred = model.predict(X_poly)

# visualization
plt.scatter(X, y)

plt.plot(X, y_pred, color="red")

plt.show()