from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib

from . import config



@dataclass
class LoadedModelState:
    model: Any = None
    loaded: bool = False
    error: Optional[str] = None
    model_uri: Optional[str] = None
    run_id: Optional[str] = None
    run_name: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)


class ModelService:
    """Loading the HW02 model from MLflow.

    Note: the model was logged as a raw joblib artifact:
        model/pipeline.pkl
    Therefore, here we download the artifact from MLflow and load it
    with joblib instead of using mlflow.sklearn.load_model().

    """

    def __init__(self) -> None:
        self.state = LoadedModelState()

    def load(self) -> None:
        try:
            artifact_path = getattr(
                config,
                "MLFLOW_MODEL_ARTIFACT_PATH",
                "model/pipeline.pkl",
            )

            model_uri = f"runs:/{config.MLFLOW_RUN_ID}/{artifact_path}"

            local_model_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri
            )

            model = joblib.load(local_model_path)

            client = MlflowClient()
            run = client.get_run(config.MLFLOW_RUN_ID)

            self.state = LoadedModelState(
                model=model,
                loaded=True,
                error=None,
                model_uri=model_uri,
                run_id=config.MLFLOW_RUN_ID,
                run_name=run.data.tags.get("mlflow.runName"),
                metrics=dict(run.data.metrics),
                params=dict(run.data.params),
                tags=dict(run.data.tags),
            )

        except Exception as exc:
            self.state = LoadedModelState(
                model=None,
                loaded=False,
                error=str(exc),
                model_uri=None,
                run_id=config.MLFLOW_RUN_ID or None,
            )

    def require_model(self):
        if not self.state.loaded or self.state.model is None:
            raise RuntimeError(self.state.error or "Model is not loaded.")
        return self.state.model

    def model_info(self) -> dict:
        return {
            "model_loaded": self.state.loaded,
            "tracking_uri": config.MLFLOW_TRACKING_URI,
            "experiment_name": config.MLFLOW_EXPERIMENT_NAME,
            "model_uri": self.state.model_uri,
            "run_id": self.state.run_id,
            "run_name": self.state.run_name,
            "dataset_version": config.DATASET_VERSION,
            "target": config.TARGET_NAME,
            "threshold": config.PREDICTION_THRESHOLD,
            "metrics": self.state.metrics,
            "params": self.state.params,
            "tags": self.state.tags,
            "error": self.state.error,
        }
