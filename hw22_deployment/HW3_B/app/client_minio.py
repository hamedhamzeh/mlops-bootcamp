"""app.client_minio — MinIO client (used only for bundle download in s3 mode)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from . import config


def get_credentials() -> dict:
    """Read MinIO connection settings from environment variables."""
    return {
        "endpoint": os.getenv("MINIO_ENDPOINT", ""),
        "access_key": os.getenv("MINIO_ACCESS_KEY", ""),
        "secret_key": os.getenv("MINIO_SECRET_KEY", ""),
        "bucket": os.getenv("MINIO_BUCKET", "hw03-bundles"),
        "prefix": os.getenv("MINIO_PREFIX", f"{config.STUDENT_USERNAME}/"),
    }


def download_bundle(target_dir: Path) -> bool:
    """Download the encoder bundle from MinIO into target_dir.

    This is only used when MODEL_SOURCE=s3.
    Returns True on success, False on any error.
    """
    try:
        from minio import Minio

        creds = get_credentials()

        endpoint = creds["endpoint"]
        access_key = creds["access_key"]
        secret_key = creds["secret_key"]
        bucket = creds["bucket"]
        prefix = creds["prefix"]

        if not endpoint or not access_key or not secret_key:
            return False

        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )

        target_dir.mkdir(parents=True, exist_ok=True)

        objects = client.list_objects(
            bucket,
            prefix=prefix,
            recursive=True,
        )

        downloaded_any = False

        for obj in objects:
            if obj.is_dir:
                continue

            object_name = obj.object_name

            if not object_name.startswith(prefix):
                continue

            rel = object_name[len(prefix):].lstrip("/")

            if not rel:
                continue

            dest = target_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            client.fget_object(bucket, object_name, str(dest))
            downloaded_any = True

        return downloaded_any

    except Exception:
        return False
