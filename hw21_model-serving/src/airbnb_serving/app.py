import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List, Optional

import joblib
import mlflow
from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from airbnb_serving.predictor import predict_batch, predict_single
from airbnb_serving.schema import ListingFeatures, PredictionResponse


load_dotenv(Path.cwd() / ".env")

@dataclass
class LoadedModelState:
    model: object | None = None
    loaded: bool = False
    error: Optional[str] = None
    model_uri: Optional[str] = None
    run_id: Optional[str] = None
    run_name: Optional[str] = None


state = LoadedModelState()


def load_model_from_mlflow() -> LoadedModelState:
    """Load the joblib model artifact from MLflow."""

    mlflow_tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://185.50.38.163:33014",
    )
    mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    mlflow_run_id = os.getenv("MLFLOW_RUN_ID")
    model_artifact_path = os.getenv(
        "MLFLOW_MODEL_ARTIFACT_PATH",
        "model/pipeline.pkl",
    )

    if not mlflow_run_id:
        raise ValueError("mlflow_run_id environment variable is required.")

    if mlflow_tracking_username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_tracking_username

    if mlflow_tracking_password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_tracking_password

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    model_uri = f"runs:/{mlflow_run_id}/{model_artifact_path}"

    local_model_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri
    )

    model = joblib.load(local_model_path)

    client = MlflowClient()
    run = client.get_run(mlflow_run_id)

    return LoadedModelState(
        model=model,
        loaded=True,
        error=None,
        model_uri=model_uri,
        run_id=mlflow_run_id,
        run_name=run.data.tags.get("mlflow.runName"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once when the FastAPI app starts."""

    global state

    try:
        state = load_model_from_mlflow()
        print(f"Model loaded successfully from {state.model_uri}")

    except Exception as exc:
        state = LoadedModelState(
            model=None,
            loaded=False,
            error=str(exc),
        )
        print(f"Model loading failed: {exc}")

    yield


app = FastAPI(
    title="Airbnb High-Demand Prediction API",
    version="0.1.0",
    description="FastAPI service for serving the HW02 Airbnb high-demand model.",
    lifespan=lifespan,
)


def get_loaded_model():
    """Return loaded model or raise API error."""

    if not state.loaded or state.model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "message": "Model is not loaded.",
                "error": state.error,
            },
        )

    return state.model


@app.get("/health")
def health():
    """Health check endpoint."""

    return {
        "status": "ok" if state.loaded else "error",
        "mlflow_run_id": state.run_id,
        "model_uri": state.model_uri,
        "run_name": state.run_name,
        "error": state.error,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ListingFeatures):
    """Predict high-demand class for one listing."""

    model = get_loaded_model()

    return predict_single(
        features=features,
        model=model,
        run_id=state.run_id or "",
    )


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch_endpoint(features_list: List[ListingFeatures]):
    """Predict high-demand class for multiple listings."""

    model = get_loaded_model()

    if not features_list:
        raise HTTPException(
            status_code=400,
            detail="Batch request must contain at least one listing.",
        )

    return predict_batch(
        features_list=features_list,
        model=model,
        run_id=state.run_id or "",
    )