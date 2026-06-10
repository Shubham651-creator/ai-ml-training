import numpy as np

rajgad = np.array([0.80, 0.90])
sinhagad = np.array([ 0.82, 0.88])

distance = np.linalg.norm(rajgad - sinhagad)
print(distance) #0.02828427124746185

### If distance is less, then the two vectors are similar. 
# In this case, the distance is very small, 
# which indicates that the two vectors are quite similar.    