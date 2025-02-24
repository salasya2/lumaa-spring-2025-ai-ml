import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk

# download the corpus
nltk.download('wordnet')

# import the lemmatizer
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('./imdb_top_1000.csv') # read the csv file

# filter to select only the required columns
df=data[['Overview','Series_Title']]




def lemmatize_text(text):
  
  # initialize the lemmatizer
  wn_lemmatizer = WordNetLemmatizer()

  # split the text into words and lemmatize them
  lemmatized_text=[wn_lemmatizer.lemmatize(word) for word in text.split()] 

  # join all the lemmatized words as a single text and return
  return ' '.join(lemmatized_text) 

#replace any empty descriptions with ''.
df['Overview']=df['Overview'].fillna('')

# convert the descriptions into lower case
df['processed_overview']=df['Overview'].apply(lambda x:x.lower())

# lemmatize the processed descriptions
df['lemmatized_overview']=df['processed_overview'].apply(lambda x:lemmatize_text(x))

def process_user_input(user_input):
  
    # convert to lower case
    user_input=user_input.lower()
    

    # lemmatize the user input
    lemmatized_user_input=lemmatize_text(user_input)

    return lemmatized_user_input  


def get_recommendations(data, item_column, description_column, user_input, top_n=5):

    # initialize the TFIDFVectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    
    #convert into numeric data using TFiDF vectorizer
    tfidf_matrix = tfidf.fit_transform(data['lemmatized_overview'])
    
    #process the user input
    processed_user_input= process_user_input(user_input)

    # vectorize the processed user input to numeric data
    user_tfidf = tfidf.transform([processed_user_input])
    
    #calculate the cosine similarity between user and descriptions in the dataset.
    cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)
    
    # pair the scores with their indices
    sim_scores = list(enumerate(cosine_sim[0]))

    # sort the scores based on similarity in descending order.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # only return top n scores so that we can have n recommendations
    sim_scores = sim_scores[:top_n]

    # create indices for the recommendations
    item_indices = [i[0] for i in sim_scores]

    #create a dataframe to return the recommendations
    recommendations = pd.DataFrame(data.iloc[item_indices][[item_column, description_column]])    
    return recommendations


def run():

  # prompt the user to give input
  user_input=input("Enter the description :\n\n")

  # call the recommendation system
  results=get_recommendations(df,'Series_Title','Overview',user_input, 5)
  print('\n\n')
  return results

print(run().reset_index(drop=True))
