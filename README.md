# BOT-DETECTION-ML-EXPERIMENT
![image](https://github.com/user-attachments/assets/c70953bb-0e7f-4384-b1c4-c96143414dc1)

This Repository is Used for Experimentation process for a BOT-Detection Model, especially leveraging Mlflow and DVC. This Document can help in navigating through the directory for the experimentation setup. Similar template can be used for other Projects. 

## Project Structure
```
bot-detection-ml-experiment/
├── data/
│   ├── raw/                         # Raw data files 
│   ├── processed/                   # Processed data files 
│   │   ├── tf_df/                   # TF-IDF processed data for different models
├── data-preparation/                # Data preparation pipeline Scripts
│   ├── src/                         # Transformation utility functions
├── models/                          # Model training scripts and saved models
│   ├── src/                         # Utility functions for model scripts
│   ├── mlruns/                      # MLflow experiment tracking directory
├── notebooks/                       # Jupyter notebooks for EDA and experimentation
├── .dvc/                            # DVC configuration files
│   ├── config                 
├── requirements.txt                 # Python dependencies

```

## Environment Setup 
Create a Virtual Environment 
```bash
python -m venv venv_name
```
Activate the virtual Environment
```bash
source venv_name/bin/activate
```
Install the Dependencies using the requirements.txt file 
```bash
pip install -r requirements.txt
```
## Initialize DVC 
DVC is initialized using the command below 
```bash
dvc init
```
This Command will create a .dvc file if the file is not Present. <br>
This dvc is configured its remote to an AWS S3 bucket, it can be easily be setup by following the Link [here](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3)
After Succefully completing the DVC setup, pull the data from the remote server using:
```bash
dvc pull
```

If you want to add new_data to dvc which can be tracked using dvc:
```bash
dvc add data/raw/raw_data.csv
```
The Above command will generate a .dvc file which store the metadata of the original file and compresses it. You can use this .dvc file to Track the data from a remote storage like s3 using:
```bash
dvc push
```

## MLflow 
To run and experiment on various models using the data from DVC, mlflow can be a great option. <br>
Start by navigating to the models directory where the python scripts, experiments, runs, model artificats are stored. First set the mlflow tracking uri using:
```bash
mlflow.set_tracking_uri("file:///mnt/e/BOT-DETECTION/BOT-DETECTION-ML-EXPERIMENT/models/mlruns")
```
This will create a mlruns/ directory. <br>

You can run the experiments using:
```bash
python models/knn.py
```

To visualise the runs, navigate to the models directory and using:
```bash
mlflow ui
```
The UI will be accessible by default at 'http://127.0.0.1:5000' in your web browser. <cd>
You Can Visualize accros the Experiments Using various Visualisations available below are few Sample how the UI can help. <br>
## MLflow UI
The below Picture is a illustration of how 40 different runs can be compared, on how the evaluation metrics vary based on different input parameters
![Screenshot (315)](https://github.com/user-attachments/assets/528076bf-3985-4193-8dda-490d6077ba95)

![Screenshot (316)](https://github.com/user-attachments/assets/4de0a7de-754d-46c6-b1b4-b01dc5d21fa5)


