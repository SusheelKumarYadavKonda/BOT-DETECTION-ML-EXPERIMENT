import mlflow 
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report 
from src.linear_model_utils import pull_and_load_data

mlflow.set_experiment("Logistic Regression Experiment")

if __name__ == '__main__':

    dvc_file_path = '/mnt/e/BOT-DETECTION/BOT-DETECTION-ML-EXPERIMENT/data/processed/tf_df.dvc'
    
    data_types = ['stemmed_tweet', 'lemmatized_tweet']

    for data_type in data_types:
        X_train, X_test, y_train, y_test = pull_and_load_data(dvc_file_path, data_type)

        with mlflow.start_run(run_name="Simple Logistic Regression with DVC Data for {data_type}"):

            model = LogisticRegression(random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions, output_dict=True)

            mlflow.log_metric("accuracy", accuracy)

            mlflow.sklearn.log_model(model, "logistic_regression_model")

            mlflow.log_text(str(report), "classification_report.txt")
            
            print(f"Logistic Regression - Accuracy: {accuracy}")

        print("Experiment completed.")

