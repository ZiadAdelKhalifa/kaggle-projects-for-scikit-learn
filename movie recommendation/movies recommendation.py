import numpy as np 
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data=pd.read_csv('movies.csv')
print(data.head())
print(data.shape)

#sellecting the relative feature for the recommendation
selected_features=['genres','keywords','tagline','cast','director']

#replacting the null value with null string
for f in selected_features:
    data[f]=data[f].fillna('')

#combining all teh five selected features
combaned_Feature=data['genres']+' '+ ['keywords']+' '+['tagline']+' '+['cast']+' '+['director']
#print(combaned_Feature)

#convereting the text data to feature vectors
vec=TfidfVectorizer()
feature_vec=vec.fit_transform(combaned_Feature)

#print(feature_vec)

#getting similarity score using cosine similarity
similarity=cosine_similarity(feature_vec)
#print(similarity)
#print(similarity.shape)

#getting the move name from the user
movie=input('enter your favorate movie :')

#creating a list with all the movie names in the dataset
list_all_title=data['title'].tolist()
#print(list_all_title)

#finding the close match for the movie name gaven by the user
find_close_match=difflib.get_close_matches(movie,list_all_title)
#print(find_close_match)
close_match=find_close_match[0]
#find the index of the movie with title

index_of_the_movie=data[data.title == close_match]['index'].values[0]
#print(index_of_the_movie)

#getting a list with similar movies
similarity_score=list(enumerate(similarity[index_of_the_movie]))
#print(len(similarity_score))

#sorting the movie based on its similar score

sorted_similar_movie=sorted(similarity_score,key=lambda x:x[1], reverse=True )
print(sorted_similar_movie)


#print the names of the semilar moveis based on the index

print("Moves suggested to you: ",'\n')

i=1
for moviee in sorted_similar_movie:
    index=moviee[0]
    title_from_index=data[data.index ==index]['title'].values[0]
    if(i<=30):
        print(i," ",title_from_index)
        i+=1
