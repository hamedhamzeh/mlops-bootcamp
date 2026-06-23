"""app.model_loader — wraps the HW3_A bundle's predict.py.

The bundle is the unit of truth. This module:
  1. Discovers the bundle on disk (env BUNDLE_DIR or ../HW3_A/bundle).
  2. Imports the BundlePredictor class.
  3. Verifies the MANIFEST.json SHAs at startup (fail-fast on corruption).
  4. Holds one global predictor instance (lifespan-managed by main.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --- Bundle discovery constants ---
# When the image is built with the bundle baked in, the structure is:
#   /app/
#     app/                    <-- this FastAPI service
#     bundle/                 <-- the bundle
#       model/                <-- model.safetensors + tokenizer files
#       predict.py            <-- BundlePredictor class
#       MANIFEST.json
#       metadata.json
#       requirements.txt

_DEFAULT_BUNDLE_IN_IMAGE = "/app/bundle"
_DEV_BUNDLE = str(
    Path(__file__).resolve().parent.parent.parent / "HW3_A" / "bundle"
)


def _resolve_bundle_dir() -> Path:
    env = os.getenv("BUNDLE_DIR", "").strip()

    # 1. ENV override
    if env:
        p = Path(env).resolve()
        if p.exists():
            return p

    # 2. Container path
    if Path(_DEFAULT_BUNDLE_IN_IMAGE).exists():
        return Path(_DEFAULT_BUNDLE_IN_IMAGE)

    # 3. Local dev path
    if Path(_DEV_BUNDLE).exists():
        return Path(_DEV_BUNDLE)

    raise FileNotFoundError(
        "Bundle not found. Checked BUNDLE_DIR, /app/bundle, and HW3_A/bundle."
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


def _verify_manifest(bundle_dir: Path) -> tuple[bool, str]:
    manifest_path = bundle_dir / "MANIFEST.json"

    if not manifest_path.exists():
        return False, "MANIFEST.json not found"

    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files", {})

    for rel, expected in files.items():

        if expected.startswith("REPLACE"):
            return False, f"Placeholder hash found for {rel}"

        file_path = bundle_dir / rel

        if not file_path.exists():
            return False, f"Missing file: {rel}"

        actual = _sha256(file_path)

        if actual != expected:
            return False, f"Hash mismatch: {rel}"

    return True, f"{len(files)} files OK"



# LoadState dataclass
@dataclass
class LoadState:
    loaded: bool = False
    error: Optional[str] = None
    bundle_dir: Optional[Path] = None
    manifest_ok: Optional[bool] = None
    manifest_msg: Optional[str] = None


# ModelService dataclass
@dataclass
class ModelService:
    state: LoadState = field(default_factory=LoadState)
    predictor: Optional[object] = None
    metadata: dict = field(default_factory=dict)

    def load(self) -> None:
        try:
            # 1. resolve bundle
            bundle_dir = _resolve_bundle_dir()
            self.state.bundle_dir = bundle_dir

            # 2. verify manifest
            ok, msg = _verify_manifest(bundle_dir)
            self.state.manifest_ok = ok
            self.state.manifest_msg = msg

            if not ok:
                raise RuntimeError(msg)

            # 3. model directory
            model_dir = bundle_dir

            if not model_dir.exists():
                raise FileNotFoundError("model/ directory missing in bundle")

            # 4. add bundle to path
            if str(bundle_dir) not in sys.path:
                sys.path.insert(0, str(bundle_dir))

            # 5. import predictor dynamically
            from predict import load_bundle  # type: ignore

            self.model, self.tokenizer = load_bundle(str(model_dir))

            # 6. load metadata
            meta_path = bundle_dir / "metadata.json"
            if meta_path.exists():
                self.metadata = json.loads(meta_path.read_text())

            # 7. mark success
            self.state.loaded = True

        except Exception as e:
            self.state.loaded = False
            self.state.error = str(e)
            raise


    def require_predictor(self):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        return self


    def info(self) -> dict:
        return {
            "bundle_loaded": self.state.loaded,
            "bundle_dir": str(self.state.bundle_dir),
            "manifest_ok": self.state.manifest_ok,
            "manifest_msg": self.state.manifest_msg,
            "metadata": self.metadata,
            "error": self.state.error,
        }
