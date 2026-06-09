# HW13 - FastAPI Model Serving

This project serves the best model from HW02 through a FastAPI application. The API loads the trained ML pipeline from MLflow and provides endpoints for single and batch prediction.

## Selected Model

Selected HW02 Run ID:

```text
ae07c048c3e1432581c322f90b0275f0
```

The model is loaded from the following MLflow artifact path:

```text
runs:/ae07c048c3e1432581c322f90b0275f0/model/pipeline.pkl
```

The pipeline was logged as a `joblib` artifact, so the API downloads the artifact from MLflow and loads it with `joblib`.

## Setup

Create and activate a virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a local `.env` file based on `.env.example`.

Example:

```env
MLFLOW_TRACKING_URI=http://185.50.38.163:33014
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password
MLFLOW_EXPERIMENT_NAME=your_experiment_name
MLFLOW_RUN_ID=ae07c048c3e1432581c322f90b0275f0
MLFLOW_MODEL_ARTIFACT_PATH=model/pipeline.pkl
PREDICTION_THRESHOLD=0.5
```

The `.env` file is not committed because it contains private credentials.

## Run the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

| Endpoint         | Method | Description                            |
| ---------------- | ------ | -------------------------------------- |
| `/`              | GET    | Basic home endpoint                    |
| `/health`        | GET    | Checks whether the model is loaded     |
| `/model-info`    | GET    | Shows MLflow run and model information |
| `/predict`       | POST   | Predicts one listing                   |
| `/predict-batch` | POST   | Predicts multiple listings             |

## Prediction

The API receives listing features as JSON, converts them into a pandas DataFrame, orders the columns according to the training feature list, and sends the data to the loaded ML pipeline.

The prediction output includes:

- predicted class
- prediction label
- probability
- threshold

The labels are:

```text
0 = not_high_demand_proxy
1 = high_demand_proxy
```

## Validation and Leakage Protection

The API validates the input schema before prediction. It rejects missing fields, wrong data types, unknown fields, and leakage/audit fields.

Forbidden fields include:

```text
listing_id
cutoff_date
dataset_version
future_calendar_days_observed_30d
future_available_days_30d
future_available_rate_30d
high_demand_proxy
```

This prevents target leakage and makes sure the API only uses valid model input features.

## Testing

Example request files are provided in the `data/` folder:

```text
valid_predict_request.json
valid_batch_request.json
bad_request_missing_field.json
bad_request_wrong_type.json
bad_request_leakage_field.json
```

Valid requests should return prediction results. Bad requests should return validation errors and should not produce predictions.

## Screenshots

The required Swagger screenshots are saved in the `screenshots/` folder:

```text
01_docs_all_endpoints.png
02_predict_schema.png
03_predict_success.png
04_predict_validation_error.png
05_batch_predict_success.png
06_model_info.png
```

## Notes

The model is loaded during application startup. When running with `--reload`, the model may be reloaded after code changes. For final testing, the app can also be run without `--reload`.
