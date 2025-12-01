# Pacemaker Worker

This worker is responsible for **training** Atomic Cluster Expansion (ACE) potentials and performing **active learning sampling** using the [Pacemaker](https://github.com/ICAMS/pacemaker) library.

## Functionality

1.  **Train**: Fits an ACE potential to a labeled dataset.
2.  **Sample**: Selects high-uncertainty candidates from a pool using MaxVol or other active learning strategies.
3.  **Direct Sample**: Selects diverse structures from a raw dataset using ACE descriptors + Clustering (e.g., BIRCH).
4.  **Validate**: Validates a trained potential against a test set.

## Docker Image

The recommended way to build and run this worker is via the root `docker-compose.yml`.

**Build:**
```bash
# From repository root
docker-compose build pace_worker
```

*Note: This worker requires GPU support (TensorFlow) for efficient training.*

## Usage

### 1. Training

```bash
docker run --rm --gpus all -v $(pwd)/data:/data pace_worker:latest \
    python /app/src/main.py train \
    --config /data/config.yaml \
    --meta-config /data/meta_config.yaml \
    --dataset /data/training_set.xyz \
    --iteration 1
```

**Arguments:**
*   `--dataset`: Path to the labeled training set (.xyz).
*   `--iteration`: Current Active Learning iteration number.
*   `--initial-potential`: (Optional) Path to a previous potential to warm-start from.
*   `--potential-yaml`: (Optional) Path to specific potential settings.
*   `--asi`: (Optional) Path to an existing Active Set Index (.asi) file.

### 2. Sampling (Active Learning)

Selects structures that the current potential is most uncertain about.

```bash
docker run --rm --gpus all -v $(pwd)/data:/data pace_worker:latest \
    python /app/src/main.py sample \
    --config /data/config.yaml \
    --meta-config /data/meta_config.yaml \
    --candidates /data/pool.xyz \
    --n_samples 50 \
    --output /data/selected.xyz
```

### 3. Direct Sampling (Clustering)

Selects diverse structures based purely on geometric descriptors.

```bash
docker run --rm --gpus all -v $(pwd)/data:/data pace_worker:latest \
    python /app/src/main.py direct_sample \
    --input /data/pool.xyz \
    --output /data/diverse_selection.xyz \
    --n_clusters 100
```

### 4. Validation

```bash
docker run --rm --gpus all -v $(pwd)/data:/data pace_worker:latest \
    python /app/src/main.py validate \
    --potential /data/output_potential.yace \
    --output /data/validation_metrics.json
```
