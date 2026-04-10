import pandas as pd 

data = {
    "Name" : ["shubham", "Ajinkya", "Anita"],
    "Family member" : ["Son","Younger brother","Mother"],
    "Data usage" :[12.3,32, 1]
}

# DataFrame = Collection of Wrapper over Numpy ndarray
# i.e DataFrame also use Vectorization
print(pd.DataFrame(data)) #Show in Table format