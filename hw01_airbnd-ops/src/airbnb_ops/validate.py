
REQUIRED_OUTPUT_COLUMNS = {
    "neighbourhood",
    "num_listings",
    "avg_price",
    "median_price",
    "avg_minimum_nights",
    "availability_365_avg",
    "total_reviews",
    "reviews_per_listing",
    "tourism_segment",
    "priority_level",
}

FORBIDDEN_PII_COLUMNS = {
    "host_name",
    "host_id",
    "reviewer_name",
    "reviewer_id",
}


def validate_summary(summary) -> None:

    if summary.empty:
        raise ValueError("Summary report is empty.")
    
    missing = REQUIRED_OUTPUT_COLUMNS - set(summary.columns)
    if missing:
        raise ValueError(f"Missing required output columns: {missing}")
    
    found_pii = FORBIDDEN_PII_COLUMNS & set(summary.columns)
    if found_pii:
        raise ValueError(f"PII detected: {found_pii}")
    
    if summary["neighbourhood"].isnull().any():
        raise ValueError("Null neighbourhood values found.")
    
    if (summary["num_listings"] <= 0).any():
        raise ValueError("num_listings must be greater than 0.")
    
    if (summary["avg_price"] < 0).any():
        raise ValueError("avg_price cannot be negative.")
    
    invalid_availability = (
        (summary["availability_365_avg"] < 0) | (summary["availability_365_avg"] > 365)
    )

    if invalid_availability.any():
        raise ValueError(
            "availability_365_avg must be between 0 and 365."
        )
    