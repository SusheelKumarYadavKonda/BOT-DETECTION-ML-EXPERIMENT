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


def initial_transformer(data):
    data["Created At"] = pd.to_datetime(data["Created At"])
    data['Hashtags'] = data['Hashtags'].fillna('')
    data['Username'] = data['Username'].str.lower()
    data['Tweet'] = data['Tweet'].str.lower() 
    data['Hashtags'] = data['Hashtags'].str.lower()
    data['Location'] = data['Location'].str.lower()
    data['Tweet'] = data['Tweet'].str.translate(str.maketrans('', '', string.punctuation))
    data['Hashtags'] = data['Hashtags'].str.translate(str.maketrans('', '', string.punctuation))

    return data

def encode_categorical(data):
    label_encoder = LabelEncoder()
    data['Verified_Encoded'] = label_encoder.fit_transform(data['Verified'])

    data.drop(['Verified'], axis=1, inplace=True)
    
    return data

def standerdize(data):
    scaler = StandardScaler()
    data['Retweet Count'] = scaler.fit_transform(data[['Retweet Count']])
    data['Mention Count'] = scaler.fit_transform(data[['Mention Count']])
    data['Follower Count'] = scaler.fit_transform(data[['Follower Count']])   
    return data


if __name__== "__main__": 
    input_path = os.path.join('/mnt/e/BOT-DETECTION/BOT-DETECTION-ML-EXPERIMENT/data/raw', 'bot_detection_data.csv')

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The file at {input_path} was not found.")
    
    output_path = '/mnt/e/BOT-DETECTION/BOT-DETECTION-ML-EXPERIMENT/data/processed/cleaned_data.csv'
    
    data = pd.read_csv(input_path)

    data = initial_transformer(data)

    data = encode_categorical(data)

    data = standerdize(data)

    data.to_csv(output_path, index=False)

    print(f"Initial Processed data is Saved to {output_path}")
   