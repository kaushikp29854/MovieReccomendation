#Movie Recommender System using Python

#Movie lens dataset
import numpy as numpy

#Get those specific columns
column_names = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('u.data', sep ='\t', names = column_names)


df.head() #userid, item_id, rating, timestamp

movie_titiles = pd.read_csv("Movie_Id_Titles")
movie_titiles.head()


df = pd.merge(df, movie_titiles, on = 'iterm_id')
df.head()

#We do some exploratory data analysis
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style('white')


#create a rating data frame with avg rating and count of ratings for the movie

#we create a data set with the mean ratings

#integrate the total count for each movie with the mean ratings

plt.figure(figsize =(10, 4))
ratings['num of ratings'].hist(bins =70)

plt.figure(figsize =(10, 4))
ratings['rating'].hist(bins =70)

#this helps us visualize if there is a correlation 
sns.jointplot(x='rating', y ='num of ratings', data = ratings, alpha =0.5)

###RECCOMENDING MOVIES

#we create a pivot table
moviemat = df.pivot_table(index = 'user_id', columnds = 'title', values = 'rating') #Nan They have not given a specif rating 

#sort the ratings based off the total number of counts

#LEts get the user ratings for two specifc movies
star_wars_rat =moviemat['Star Wars (1977)']
liwar_ratin =moviemat['Liar Liar (1997)']

similar_to_starwars = moviemat.corrwith(star_wars_rat)
similar_to_liar = moviemat.corrwith(liwar_ratin)

#create a data set using the similawr to vairbales

#Have a threshold fo rthe number of ratings(only consider movies with more than 100 ratings)

############################################################
######Weighted Hybrid Techniquw for Reccomendation systtem####

import pandas as pd
import numpy as np 
import seaborn as sns 

#READ IN THE DATA
credits = pd.read_csv("tmdb_5000_credits.csv") #movieid, title, cast, crew
movies_df = pd.read_csv("tmdb_5000_credits.csv") 

#print the shape out for both of the data sets
credit_column_renamed = credits.rename(index = str, columns = {"movie_id": "id"}) #rename movie_id to id to be able tp merge the two datasets

#combine the credits and movie data set, combining with respec to id, because movie_id and id are very similar
movies_df_merge = movies_df.merge(credits_column_renamed, on = "id")

movies_df_merge.head(5)

#Drop featues that are not important to the reccomendation

#We use the weighted average formula  and set varaibles to each of the components

#Sort the weighted average column
movie_sorted_ranking = movies_df.sort_values('weight_average', ascending =False)

#we can plot the weighted average
weight_average = movie_sorted_ranking.sort_values('weight_average', ascending= False)
axis1 = sns.barplot(x= weight_average['weighted_average'].head(10), y = weight_average)

######################################################################
#######Building content based reccomendation using cosine similarity ############################

import pandas as pd
from rake_nltk import Rake
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


diff['key_words']= ''

for index, row in df.iterrows():
	plot = row['Plot']

	#instantiating rake, by default it uses english stop Words from nltk
	#and discard all punctuation characters
	r = Rake()

	#Extracting the words by passing the text
	r.extract_keywords_from_text(plot)

	#getting the dictionary with key words and thier scores
	key_words_dict_scores = r.get_word_degrees()

	#assigning the key words to the new column
	row['key_words'] = list(key_words_dict_scores.keys())

#dropping the plot column
df.drop(columns = ['Plot'], inplace = True)

#Reset the indices to the titile column
df.set_index('Title', inplace = True)
df.head() #we have now transformed our plot into a set of keywords


#combine into a bag of words(vectorizing each sentence)
df['bag_of_words'] = ''
columns = df.columns #store the list of all the columsn
for index, row in df.iterrows():
	words = ''
	for col in columns:
		if col != 'Director': #we ignore the director columns
			words = words + ' '.join(row[col])+ ' ' #we combine all other column wrods
		else:
			words = words +row[col] + ''
			
	row['bag_of_words'] = words

#we store all the words across the columns into the 'bag_of_words' column
df.drop(columns = [col for col in df.columns if col ! = 'bag_of_words'], inplace = True)

#The bag of words column is a representation of all the columns

#Instantiartin and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])

#Creating a seried for mpvie titles so they are associated to an ordered numerical
#list I will use for later to match the indiced

indices = pd.Series(df.index)
indices[:5]

cosine_sim = cosine_similarity(count_matrix, count_matrix)
cosine_sim.shape

#function that takes in movie titles as input and returns the top 10 reccomended movies
def reccomendations(title, cosine_sim = cosine_sim):
	reccomended_movies = []

	#getting the index of the movie that matches the title
	idx = indices[indices == title].index(0)

	#creating a series with the similarity scores in descending order
	score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

	#getting the indexes of the 10 most similar movies
	top_10_indexes = list(score_series.iloc[1:11].index)

	#populating the list with the titles of the 10 best matching movies
	for i in top_10_indexes:
		reccomended_movies.append(list(df.index)[i])

	return reccomended_movies

#Sample input to our function
reccomendations('Fargo')


