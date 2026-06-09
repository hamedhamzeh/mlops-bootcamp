from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from fastapi import HTTPException, status

from . import config
from .schemas import ListingFeatures, PredictionResponse


def records_to_dataframe(records: Iterable[ListingFeatures]) -> pd.DataFrame:
    """Convert validated API payloads into the exact DataFrame expected by the model."""
    rows = [record.model_dump() for record in records]
    df = pd.DataFrame(rows)
    
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "At least one record is required."},
        )

    expected_cols = set(config.EXPECTED_FEATURE_COLUMNS)
    received_cols = set(df.columns)
    forbidden_cols = set(config.FORBIDDEN_FIELDS)

    # reject unknown fields and forbidden leakage fields.
    forbidden_received = sorted(received_cols.intersection(forbidden_cols))
    if forbidden_received:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Request contains forbidden leakage or audit fields.",
                "forbidden_fields": forbidden_received,
            },
        )
    
    extra_cols = sorted(received_cols - expected_cols)
    if extra_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Request contains unknown fields.",
                "unknown_fields": extra_cols,
            },
        )
    
    # check missing fields against config.EXPECTED_FEATURE_COLUMNS.
    missing_cols = sorted(expected_cols - received_cols)
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Request is missing required model feature fields.",
                "missing_fields": missing_cols,
            },
        )
    
    model_df = df[config.EXPECTED_FEATURE_COLUMNS].copy()

    model_df = model_df.rename(
        columns={
            "host_is_superhost": "is_superhost",
        }
    )

    return model_df


def predict_records(model, records: List[ListingFeatures]) -> List[PredictionResponse]:
    """Run model prediction and return API responses."""
    X = records_to_dataframe(records)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        classes = getattr(model, "classes_", None)

        if classes is None and hasattr(model, "steps"):
            final_estimator = model.steps[-1][1]
            classes = getattr(final_estimator, "classes_", None)

        if classes is not None and 1 in classes:
            positive_class_index = list(classes).index(1)
        else:
            positive_class_index = 1

        positive_probabilities = probabilities[:, positive_class_index]

    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Loaded model does not support predict_proba.",
        )

    predictions = [
        int(probability >= config.PREDICTION_THRESHOLD)
        for probability in positive_probabilities
    ]

    return [
        PredictionResponse(
            prediction=prediction,
            prediction_label=(
                config.POSITIVE_LABEL
                if prediction == 1
                else config.NEGATIVE_LABEL
            ),
            probability=float(probability),
            threshold=config.PREDICTION_THRESHOLD,
        )
        for prediction, probability in zip(predictions, positive_probabilities)
    ]
