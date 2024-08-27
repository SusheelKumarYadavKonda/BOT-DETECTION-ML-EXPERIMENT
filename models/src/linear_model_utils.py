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
import shutil
import mlflow
import json 

def pull_and_load_data(dvc_folder_path, data_type):
    # Pull the data using DVC
    subprocess.run(["dvc", "pull", dvc_folder_path], check=True)
    
    # Remove the '.dvc' extension from the folder path to get the actual data directory
    data_dir = dvc_folder_path.replace('.dvc', '')
    
    # Define the paths to the data files based on the data type
    X_train_path = os.path.join(data_dir, f'{data_type}_X_train.npy')
    X_test_path = os.path.join(data_dir, f'{data_type}_X_test.npy')
    y_train_path = os.path.join(data_dir, f'{data_type}_y_train.npy')
    y_test_path = os.path.join(data_dir, f'{data_type}_y_test.npy')
    
    # Load the data files
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    
    return X_train, X_test, y_train, y_test

def get_traditional_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='linear'),  # Support Vector Machine
        "NaiveBayes": MultinomialNB(),  # Naive Bayes
        "KNN": KNeighborsClassifier(),  # K-Nearest Neighbors
        "DecisionTree": DecisionTreeClassifier(),  # Decision Tree
        "RandomForest": RandomForestClassifier(n_estimators=100),  # Random Forest
        "GradientBoosting": GradientBoostingClassifier(),  # Gradient Boosting
        "RidgeClassifier": RidgeClassifier(),
        "LassoClassifier": Lasso()  # Lasso Regression
    }
    return models
def run_experiment(model_name, model, X_train, y_train, X_test, y_test, data_type):
    with mlflow.start_run(run_name=f"{model_name}_{data_type}"):
        # Log basic parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("data_type", data_type)
        
        # Log hyperparameters specific to the model (if any)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metrics({f"{label}_f1_score": report[label]["f1-score"] for label in report if label.isdigit()})
        
        # Log model
        mlflow.sklearn.log_model(model, model_name)
        
        # Log artifacts: confusion matrix and classification report
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