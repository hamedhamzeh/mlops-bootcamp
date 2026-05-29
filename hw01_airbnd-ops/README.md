# Setup & Test Instructions

Clone the repo and move to the hw01_airbnd-ops directory:

```bash
git clone "https://github.com/hamedhamzeh/mlops-bootcamp.git"
cd hw01_airbnd-ops
```

# Local Python Execution

Install the package in editable mode:

```bash
pip install -e .
```

Run the pipeline:

```bash
airbnb-ops run
```

---

# Docker Execution

Build the Docker image:

```bash
docker compose build
```

Run the pipeline container:

```bash
docker compose run --rm airbnb-ops
```

---

# DVC Reproducibility

Initialize DVC:

```bash
dvc init
```

Reproduce the pipeline:

```bash
dvc repro
```

---
