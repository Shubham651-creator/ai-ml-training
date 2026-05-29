import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# user-item matrix
data = {
    "Movie_A": [5,5,1],
    "Movie_B": [4,4,2],
    "Movie_C": [0,5,1]
}

users = ["Shubham", "Omkar", "Sushil"]

df = pd.DataFrame(data, index=users)

print(df)

# similarity
similarity = cosine_similarity(df)

# The output matrix represents cosine similarity scores between 
# users based on their movie ratings. Values close to 1 indicate highly 
# similar viewing behavior, which helps the recommendation system 
# suggest items liked by similar users.
print(similarity)