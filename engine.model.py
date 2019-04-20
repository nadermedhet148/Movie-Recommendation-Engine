import pandas as pd
import numba as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#load data set
data = pd.read_csv("./movie_dataset.csv")
# select features
features = ['keywords' , 'cast' , 'genres' , 'director']
#remove null data
for feature in features :
    data[feature] = data[feature].fillna('')
# combine features as one string
def combine_rows( row ):
    return row['keywords'] + '' + row['cast'] + '' +row['genres'] + '' +row['director'] 
# create new feature 
data['combined_features'] = data.apply(combine_rows , axis = 1)

# create count vectors from new feature
cv = CountVectorizer()
count_matrix = cv.fit_transform(data['combined_features'])
# get cosine_similarity from count_matrix
cos_sim = cosine_similarity(count_matrix)

#USE MODEL as test 
#get one of movies index
movie_index = data[data.title == "Avatar"]["index"].values[0]
#get similer of selected movie
similar_movies =  list(enumerate(cos_sim[movie_index]))
# sort similer movie
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)
# print first 10 similer movie names
i=0
for element in sorted_similar_movies:
		print (data[data.index == element[0]]["title"].values[0])
		i=i+1
		if i>10:
                 break
