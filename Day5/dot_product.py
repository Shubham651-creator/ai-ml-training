import numpy as np

# vector
x = np.array([25,3])

# weights
w = np.array([2000, 5000])

# Dot product
## prediction = x.w +b
result = np.dot(x,w)

print(result)