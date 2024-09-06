# linear-regression

This project demonstrates the use of various machine learning models and hyperparameter tuning techniques using a rice classification dataset. It includes model evaluation, training, and deployment using FastAPI.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Model Training and Hyperparameter Tuning](#model-training-and-hyperparameter-tuning)
- [API Usage](#api-usage)

## Introduction

The project performs the following tasks:
- Data preprocessing and model training using Linear Regression and XGBoost.
- Hyperparameter tuning using `GridSearchCV` and `RandomizedSearchCV`.
- Model persistence with `joblib`.
- Serving the trained model through a FastAPI application.

## Requirements

Ensure you have the following dependencies installed:

```bash
pandas
scikit-learn
xgboost
fastapi
uvicorn
joblib
```

Install the required packages:

```bash
pip install pandas scikit-learn xgboost fastapi uvicorn joblib
```

## Dataset

The dataset used in this project is the Rice Classification Dataset, which you can download from Kaggle. Make sure to adjust the `file_path` in `main.py` to your local file path after downloading the dataset.

Example of dataset download using Kaggle API:

```python
kaggle.api.dataset_download_files('mssmartypants/rice-typeclassification', path='D:/code-test/1/', unzip=True)
```

## How to Run

1. **Train the model**:
    - Edit the `file_path` variable in `main.py` to point to your dataset location.
    - Run `main.py` to perform model training and hyperparameter tuning.
    
    ```bash
    python main.py
    ```

    The model will be saved as `model.pkl`.

2. **Start the FastAPI server**:
    - Run `run_model.py` to start the FastAPI server.

    ```bash
    uvicorn run_model:app --reload
    ```

3. **Test the API**:
    You can use tools like Postman or `curl` to send POST requests to the API.

    Example `curl` request:

    ```bash
    curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
      "id": 1,
      "Area": 1234,
      "MajorAxisLength": 200.5,
      "MinorAxisLength": 150.3,
      "Eccentricity": 0.3,
      "ConvexArea": 2500,
      "EquivDiameter": 45.6,
      "Extent": 0.7,
      "Perimeter": 125.6,
      "Roundness": 0.8,
      "AspectRation": 1.5
    }'
    ```

    The response will return the `id` and `prediction`.

## Model Training and Hyperparameter Tuning

The `main.py` script performs the following:

1. Loads the dataset and splits it into training and testing sets.
2. Trains a Linear Regression model and evaluates its performance.
3. Trains an XGBoost classifier and evaluates its accuracy.
4. Hyperparameter tuning with:
   - `GridSearchCV` for exhaustive search over specified parameters.
   - `RandomizedSearchCV` for a randomized search over parameters.

After the best model is found, it's saved as `model.pkl`.

## API Usage

The `run_model.py` script uses FastAPI to serve predictions based on the trained model. You can send data in the following format:

```json
{
  "id": 1,
  "Area": 1234,
  "MajorAxisLength": 200.5,
  "MinorAxisLength": 150.3,
  "Eccentricity": 0.3,
  "ConvexArea": 2500,
  "EquivDiameter": 45.6,
  "Extent": 0.7,
  "Perimeter": 125.6,
  "Roundness": 0.8,
  "AspectRation": 1.5
}
```

The response will provide the predicted class for the given data.
