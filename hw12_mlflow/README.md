# MLflow Experiment Tracking

This project demonstrates experiment tracking and model comparison using MLflow.

Multiple models were trained for a binary classification task and all runs were logged to a remote MLflow tracking server. The goal was to compare model performance, detect data leakage, and select a production candidate.

## What the notebook does

- Connects to the MLflow tracking server
- Trains several baseline and machine learning models
- Logs parameters, metrics, tags, and artifacts to MLflow
- Detects and flags a model with data leakage
- Compares all experiment runs
- Selects the best clean model as the production candidate

## Models evaluated

- Dummy baseline
- Logistic Regression
- Logistic Regression (class balanced)
- Logistic Regression (threshold tuned)
- Random Forest

## Result

The Random Forest model achieved the best performance and was selected as the production candidate based on the evaluation metrics tracked in MLflow.

## Requirements

Install dependencies with:
```text
pip install -r requirements.txt
