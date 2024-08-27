import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from src.linear_model_utils import pull_and_load_data

mlflow.set_experiment("Initial Experiment")

models = {
    "LogisticRegression": LogisticRegression(max_iter=200, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42), 
    "SVM": SVC(kernel='linear', random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "NaiveBayes": GaussianNB(),
    "KNeighbors": KNeighborsClassifier(n_neighbors=5)
}

# Main script
if __name__ == '__main__':
    dvc_file_path = '/mnt/e/BOT-DETECTION/BOT-DETECTION-ML-EXPERIMENT/data/processed/tf_df.dvc'
    data_types = ['stemmed_tweet', 'lemmatized_tweet']

    for data_type in data_types:
        X_train, X_test, y_train, y_test = pull_and_load_data(dvc_file_path, data_type)
        
        for model_name, model in models.items():
            with mlflow.start_run(run_name=f"{model_name} with {data_type}"):
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average='weighted')
                recall = recall_score(y_test, predictions, average='weighted')
                f1 = f1_score(y_test, predictions, average='weighted')

                precision_0 = precision_score(y_test, predictions, pos_label=0)
                precision_1 = precision_score(y_test, predictions, pos_label=1)
                recall_0 = recall_score(y_test, predictions, pos_label=0)
                recall_1 = recall_score(y_test, predictions, pos_label=1)
                f1_0 = f1_score(y_test, predictions, pos_label=0)
                f1_1 = f1_score(y_test, predictions, pos_label=1)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision_weighted", precision)
                mlflow.log_metric("recall_weighted", recall)
                mlflow.log_metric("f1_score_weighted", f1)

                mlflow.log_metric("precision_0", precision_0)
                mlflow.log_metric("precision_1", precision_1)
                mlflow.log_metric("recall_0", recall_0)
                mlflow.log_metric("recall_1", recall_1)
                mlflow.log_metric("f1_score_0", f1_0)
                mlflow.log_metric("f1_score_1", f1_1)

                mlflow.sklearn.log_model(model, model_name)

                report = classification_report(y_test, predictions, output_dict=True)
                mlflow.log_text(str(report), "classification_report.txt")

                print(f"{model_name} - Accuracy: {accuracy}, Precision (0): {precision_0}, Precision (1): {precision_1}")
    
    print("All experiments completed.")