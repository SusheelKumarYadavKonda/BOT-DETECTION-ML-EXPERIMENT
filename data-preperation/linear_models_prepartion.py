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
from src.processing_utils import tf_df, pull_data, save_split_data

if __name__ == '__main__' :

    dvc_file_path = '/mnt/e/BOT-DETECTION/BOT-DETECTION-ML-EXPERIMENT/data/processed/cleaned_data.csv.dvc'
    input_path = '/mnt/e/BOT-DETECTION/BOT-DETECTION-ML-EXPERIMENT/data/processed/cleaned_data.csv'
    output_folder = '/mnt/e/BOT-DETECTION/BOT-DETECTION-ML-EXPERIMENT/data/processed/tf_df'

    data = pull_data(dvc_file_path, input_path)
    if data is not None:
        methods = ['Stemmed Tweet', 'Lemmatized Tweet']
        for method in methods:
            X, y = tf_df(data, method=method)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            save_split_data(output_folder, X_train, X_test, y_train, y_test, method)
        print("All datasets have been saved successfully.")
    else:
        print("Failed to load data. Exiting program.")
    
