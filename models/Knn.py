import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from src.linear_model_utils import pull_and_load_data
from itertools import product

mlflow.set_experiment("KNeighbors Hyperparameter Tuning")

param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

if __name__ == '__main__':
    dvc_file_path = '/mnt/e/BOT-DETECTION/BOT-DETECTION-ML-EXPERIMENT/data/processed/tf_df.dvc'

    data_types = ['stemmed_tweet']

    for data_type in data_types:
        X_train, X_test, y_train, y_test = pull_and_load_data(dvc_file_path, data_type)
        
        for n_neighbors, weights, metric in product(param_grid['n_neighbors'], param_grid['weights'], param_grid['metric']):
            with mlflow.start_run(run_name=f"KNeighbors with {data_type} - n_neighbors={n_neighbors}, weights={weights}, metric={metric}"):
                model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

                model.fit(X_train, y_train)
                
                predictions = model.predict(X_test)

                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average='weighted')
                recall = recall_score(y_test, predictions, average='weighted')
                f1 = f1_score(y_test, predictions, average='weighted')

                mlflow.log_param("n_neighbors", n_neighbors)
                mlflow.log_param("weights", weights)
                mlflow.log_param("metric", metric)
                

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision_weighted", precision)
                mlflow.log_metric("recall_weighted", recall)
                mlflow.log_metric("f1_score_weighted", f1)

                mlflow.sklearn.log_model(model, "knn_model")

                report = classification_report(y_test, predictions, output_dict=True)
                mlflow.log_text(str(report), "classification_report.txt")

                print(f"KNeighbors - Accuracy: {accuracy}, n_neighbors={n_neighbors}, weights={weights}, metric={metric}")

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/knn_model"
        mlflow.register_model(model_uri, "KNeighborsClassifierModel")
    
    print("KNeighbors hyperparameter tuning completed.")