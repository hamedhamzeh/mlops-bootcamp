from typing import List
import pandas as pd

from airbnb_serving.schema import ListingFeatures, PredictionResponse

FEATURE_COLS = [
    'room_type', 'property_type', 'neighbourhood_name',
    'accommodates', 'bedrooms', 'beds', 'bathrooms', 'listing_price',
    'minimum_nights', 'maximum_nights', 'instant_bookable', 'is_superhost',
    'host_listing_count', 'total_reviews_before_cutoff', 'unique_reviewers_before_cutoff',
    'avg_comment_len_before_cutoff', 'max_comment_len_before_cutoff',
    'days_since_last_review', 'available_days_last_90d', 'available_rate_last_90d',
    'avg_minimum_nights_calendar_last_90d', 'avg_maximum_nights_calendar_last_90d',
    'available_days_last_30d', 'available_rate_last_30d',
    'avg_minimum_nights_calendar_last_30d', 'avg_maximum_nights_calendar_last_30d',
]



def _features_to_dataframe(features_list: List[ListingFeatures]) -> pd.DataFrame:
    """Convert validated Pydantic feature rows into a model-ready DataFrame."""

    rows = [
        features.model_dump()
        for features in features_list
    ]

    df = pd.DataFrame(rows)
    # df = df.rename(columns={"host_is_superhost": "is_superhost"})


    # Keep the exact feature order used during training.
    return df[FEATURE_COLS]



def _get_high_demand_probabilities(model, X: pd.DataFrame) -> list[float]:
    """Return probability of class 1 if predict_proba is available."""

    if not hasattr(model, "predict_proba"):
        return [float("nan")] * len(X)

    probabilities = model.predict_proba(X)

    if probabilities.shape[1] == 1:
        return [float(probabilities[i, 0]) for i in range(len(X))]

    return [float(probabilities[i, 1]) for i in range(len(X))]

def predict_single(
    features: ListingFeatures,
    model,
    run_id: str,
) -> PredictionResponse:
    """Predict high-demand class for one listing."""

    X = _features_to_dataframe([features])

    predictions = model.predict(X)
    probabilities_high_demand = _get_high_demand_probabilities(model, X)

    return PredictionResponse(
        listing_id=None,
        prediction=int(predictions[0]),
        probability_high_demand=float(probabilities_high_demand[0]),
        model_run_id=str(run_id),
    )



def predict_batch(
    features_list: List[ListingFeatures],
    model,
    run_id: str,
) -> List[PredictionResponse]:
    """Predict high-demand class for multiple listings in one model call."""

    X = _features_to_dataframe(features_list)

    predictions = model.predict(X)
    probabilities_high_demand = _get_high_demand_probabilities(model, X)

    responses = []

    for prediction, probability in zip(predictions, probabilities_high_demand):
        responses.append(
            PredictionResponse(
                listing_id=None,
                prediction=int(prediction),
                probability_high_demand=float(probability),
                model_run_id=str(run_id),
            )
        )

    return responses