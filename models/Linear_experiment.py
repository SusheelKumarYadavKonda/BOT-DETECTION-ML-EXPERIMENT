import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import subprocess
import json 

mlflow.set_tracking_uri(uri ="./my_runs")
# Function to pull data from DVC and load it
def pull_and_load_data(dvc_folder_path, data_type):
    subprocess.run(["dvc", "pull", dvc_folder_path], check=True)
    data_dir = dvc_folder_path.replace('.dvc', '')

    X_train_path = os.path.join(data_dir, f'{data_type}_X_train.npy')
    X_test_path = os.path.join(data_dir, f'{data_type}_X_test.npy')
    y_train_path = os.path.join(data_dir, f'{data_type}_y_train.npy')
    y_test_path = os.path.join(data_dir, f'{data_type}_y_test.npy')

    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    
    return X_train, X_test, y_train, y_test


# Function to get traditional machine learning models
def get_traditional_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='linear'),
        "NaiveBayes": MultinomialNB(),
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "GradientBoosting": GradientBoostingClassifier(),
        "RidgeClassifier": RidgeClassifier(),
        "LassoClassifier": Lasso()
    }
    return models

# Function to run an experiment on a given model and log results with MLflow
def run_model_experiment(model_name, model, X_train, y_train, X_test, y_test, data_type):
    with mlflow.start_run(nested=True):
        mlflow.set_tag("data_type", data_type)
        mlflow.log_param("model_type", model_name)
        
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, model_name)
        
        confusion_matrix_path = f"confusion_matrix_{data_type}_{model_name}.json"
        classification_report_path = f"classification_report_{data_type}_{model_name}.json"
        
        with open(confusion_matrix_path, "w") as f:
            json.dump(confusion_matrix(y_test, predictions).tolist(), f)
        mlflow.log_artifact(confusion_matrix_path)
        
        with open(classification_report_path, "w") as f:
            json.dump(report, f)
        mlflow.log_artifact(classification_report_path)
        
        print(f"{model_name} - Accuracy on {data_type}: {accuracy}")
        print(report)


# Main function to run all traditional models in a single overarching run
if __name__ == "__main__":
    

    # Define the types of data
    dvc_folder_path = '/mnt/e/BOT-DETECTION/BOT-DETECTION-ML-EXPERIMENT/data/processed/tf_df.dvc'
    data_types = ['stemmed_tweet', 'lemmatized_tweet']

    # Iterate over each data type
    for data_type in data_types:
        X_train, X_test, y_train, y_test = pull_and_load_data(dvc_folder_path, data_type)
        
        # Start a single overarching MLflow run
        with mlflow.start_run(run_name=f"Traditional ML Experiments on {data_type}"):
            models = get_traditional_models()
            for model_name, model in models.items():
                run_model_experiment(model_name, model, X_train, y_train, X_test, y_test, data_type)