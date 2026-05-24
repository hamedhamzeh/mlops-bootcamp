import hashlib
import pandas as pd


DIRECT_PII_COLUMNS = ["host_name"]



def pseudonymize_value(value, salt="qbc12"):
    """
    stable SHA256 pseudonymizer.
    """
    raw = f"{salt}_{value}"

    return hashlib.sha256(raw.encode()).hexdigest()



def handle_pii(df):
    """
    Remove direct PII and pseudonymize host IDs.
    """

    df = df.copy()
    df = df.drop(columns=DIRECT_PII_COLUMNS, errors="ignore")
    df["host_key"] = df["host_id"].apply(pseudonymize_value)
    df = df.drop(columns=["host_id"], errors="ignore")

    return df 