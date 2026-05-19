import numpy as np

# Matrix Addition
a = [0,3,6],[1,4,7],[2,5,8] # List1 (linked list)
aMatix = np.array(a)        # n-dimentional Array
b= [7,6,5]                  # List2 (linked list)
bMatrix = np.array(b)       # ndarray

# print(np.array(a+b)) - Add to two lists
print(np.array(aMatix+bMatrix))
print(np.mean(aMatix+bMatrix))
print(np.shape(aMatix+bMatrix))

# 2D Matrix
matrix = np.array([[1,2,3],[4,5,6]])
print(matrix.shape)   # (2,3)

# Vectorization >>>> Normal Loops on Matrix
print("Addition by 1 on a: \n",aMatix + 1)
print("Product by 10: \n", aMatix * 10)

print("Addition by 1 on b: \n",bMatrix + 1)
# Normal Loops on Matrix
print("Normal Loops on Matrix")
result = []
for x in b:
    result.append(x+1)
    
print(result)