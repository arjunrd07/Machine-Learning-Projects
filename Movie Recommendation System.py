#!/usr/bin/env python
# coding: utf-8

# # Project On Movie Recommendation System

# ## Importing Libaries

# In[1]:


import numpy as np
import pandas as pd
import difflib # used to get close matches
from sklearn.feature_extraction.text import TfidfVectorizer #Convert the Text data to feature Vector
from sklearn.metrics.pairwise import cosine_similarity # Used to Find the Similarity Score


# In[2]:


movies_data = pd.read_csv("movies.csv")


# In[3]:


movies_data.head()


# In[4]:


movies_data.shape


# In[5]:


selected_features = ['genres','keywords','tagline','cast','director']


# Replacing the null valuess with null string

# In[6]:


movies_data.isnull().sum()


# In[7]:


for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')


#  Combining all the 6 selected features

# In[8]:


combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']


# In[9]:


print(combined_features)


# converting the text data to feature vectors

# In[10]:


vectorizer = TfidfVectorizer()


# In[11]:


feature_vectors = vectorizer.fit_transform(combined_features)


# In[12]:


print(feature_vectors)


# Codine Similarity -- IUsed to Find Similarity in the Data

# In[13]:


similarity = cosine_similarity(feature_vectors)


# In[14]:


print(similarity)
print(similarity.shape)


# In[15]:


movie_name = input(' Enter your favourite movie name : ')


# In[16]:


list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)


# ## Finding the close match for the movie name given by the user

# In[17]:


find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)
close_match = find_close_match[0]


# finding the index of the movie with title

# In[18]:


index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)


# In[19]:


# getting a list of similar movies

similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[20]:


sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)


# In[21]:


# print the name of similar movies based on the index

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<=10):
    print(i, '.',title_from_index)
    i+=1


# In[26]:


movie_name = input(' Enter your favourite movie name : ')

list_of_all_titles = movies_data['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<=15):
    print(i, '.',title_from_index)
    i+=1

