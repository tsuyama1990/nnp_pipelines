# Generation Worker

This worker handles structure generation and filtering using **MACE** (Massively Atomic Chemical Expansion), a foundational Machine Learning Force Field. It is primarily used during the "Seed Generation" phase to create an initial diverse dataset.

## Functionality

1.  **Generate**: Creates candidate structures (e.g., random crystals, defects) and relaxes them using a pre-trained MACE model to ensure they are physically reasonable.
2.  **Filter**: Filters structures based on energy or force criteria.

## Docker Image

**Build Command:**
```bash
# From repository root
docker build -t gen_worker:latest -f workers/gen_worker/Dockerfile .
```

*Note: This worker requires GPU support (NVIDIA Drivers).*

## Usage

### 1. Generation

Generates structures based on configuration settings.

```bash
docker run --rm --gpus all -v $(pwd)/data:/data gen_worker:latest \
    python /app/src/main.py generate \
    --config /data/config.yaml \
    --output /data/generated_structures.xyz
```

**Arguments:**
*   `--config`: Path to config file.
*   `--output`: Output path for generated XYZ file.

### 2. Filtering

Filters an input XYZ file using MACE (e.g., removing high-force structures).

```bash
docker run --rm --gpus all -v $(pwd)/data:/data gen_worker:latest \
    python /app/src/main.py filter \
    --input /data/raw_structures.xyz \
    --output /data/filtered_structures.xyz \
    --model medium \
    --fmax 100.0
```

**Arguments:**
*   `--input`: Input XYZ file.
*   `--output`: Output XYZ file.
*   `--model`: MACE model size (default: `medium`).
*   `--fmax`: Maximum force tolerance (eV/A). Structures with forces higher than this may be rejected or capped.
