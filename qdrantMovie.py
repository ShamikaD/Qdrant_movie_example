from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import pandas as pd

client = QdrantClient("localhost", port = 6333)
name = "movieRecs"

# #create a collection 
# client.create_collection(
#     collection_name = name,
#     vectors_config = models.VectorParams(size = 50, distance = models.Distance.COSINE)
# )

df = pd.read_csv("movies.csv", names = ["id", "rating_1", "rating_2", "title", "language", "rated_R", "genres", "avg_user_rating", "popularity"])
indices = df["id"]
df = df[["language", "rating_1", "rating_2", "popularity", "rated_R", "avg_user_rating"]]

#creating onehot encodings of categorical feature
df = df.join(pd.get_dummies(df['language']))
df = df.drop(["language"], axis = 1)

#making the df into a numpy array of floats to pass into the collection
data = np.asfarray(df.to_numpy())

#add data to the collection
client.upsert(
    collection_name = name,
    points = models.Batch(
        ids = indices,
        vectors = data.tolist()
    )
)

#find the 2 most similar movies to to the first movie in the df
first_mov = data[0][:]
print("ID of first movie: ", indices[0])
print(client.search(
    collection_name = name,
    query_vector = first_mov,
    limit = 2
))

#The most similar movies are the movie itself and the next movie!
