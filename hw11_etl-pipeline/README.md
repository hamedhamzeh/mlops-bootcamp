# Airbnb ETL Pipeline

This project builds a clean ML-ready dataset from the Airbnb PostgreSQL database.

The notebook creates one row per listing and uses a fixed cutoff date.  
All features are built only from past data, while the target is built from future calendar availability.

## What the pipeline does

- Connects to the PostgreSQL database
- Inspects tables and columns
- Checks data quality
- Audits and removes PII columns
- Builds listing, review, and calendar features
- Creates a high-demand proxy target
- Validates the final dataset
- Saves versioned outputs as CSV, Parquet, metadata, validation report, and PII audit

## Output

The generated files are saved in:

```text
data/features/
```

## Requirements

Install the dependencies with:

```text
pip install -r requirements.txt
```

## Notes

Database credentials should be set as environment variables.
