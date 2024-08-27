# BOT-DETECTION-ML-EXPERIMENT
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
