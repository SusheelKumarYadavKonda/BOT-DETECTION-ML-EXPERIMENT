import subprocess
import pandas as pd
import re
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import os  
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def Tokenizing(data):
    #Tokenizing the Tweet
    data['Tokenized Tweet'] = data['Tweet'].apply(nltk.word_tokenize)

    # Stop Word Removal
    stop_words = set(stopwords.words('english'))
    data['Tokenized Tweet'] = data['Tokenized Tweet'].apply(lambda x: [word for word in x if word not in stop_words])
    
    return data 

def stemming_tweet(data):
    # Stemming
    data = Tokenizing(data) 
    stemmer = PorterStemmer()
    data['Stemmed Tweet'] = data['Tokenized Tweet'].apply(lambda x: [stemmer.stem(word) for word in x]) 
    data['Stemmed Tweet'] = data['Stemmed Tweet'].apply(lambda x: ' '.join(x))

    return data

def lemmetization(data):
    # Lemmetization
    data = Tokenizing(data) 
    lemmatizer = WordNetLemmatizer()
    data['Lemmatized Tweet'] = data['Tokenized Tweet'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])  
    data['Lemmatized Tweet'] = data['Lemmatized Tweet'].apply(lambda x: ' '.join(x))

    return data 

def tf_df(data, method):

    vectorizer = TfidfVectorizer(max_features=5000)
    if method == 'Stemmed Tweet':
        data = stemming_tweet(data)
    else:
        data = lemmetization(data)

    X_tf =  vectorizer.fit_transform(data[method].fillna('')).toarray()
    y = data['Bot Label'].values 

    return X_tf, y

def pull_data(dvc_file_path, data_file_path):
    try:
        # Pull the data using the .dvc file
        subprocess.run(["dvc", "pull", dvc_file_path], check=True)
        print(f"Data pulled successfully using {dvc_file_path}")
        
        # Check if the data file exists
        if os.path.exists(data_file_path):
            # Load the data
            data = pd.read_csv(data_file_path)
            print(f"Data loaded successfully from {data_file_path}")
            return data
        else:
            print(f"Error: Data file {data_file_path} not found.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error pulling data: {e}")
        return None
    
def save_split_data(output_folder, X_train, X_test, y_train, y_test, method):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create method-specific filenames
    base_filename = method.replace(" ", "_").lower()
    
    np.save(os.path.join(output_folder, f'{base_filename}_X_train.npy'), X_train)
    np.save(os.path.join(output_folder, f'{base_filename}_X_test.npy'), X_test)
    np.save(os.path.join(output_folder, f'{base_filename}_y_train.npy'), y_train)
    np.save(os.path.join(output_folder, f'{base_filename}_y_test.npy'), y_test)
    
    print(f"Data for {method} saved to {output_folder}")
