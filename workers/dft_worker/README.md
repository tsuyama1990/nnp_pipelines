# DFT Worker

This worker is responsible for performing First-Principles calculations to create labeled datasets for training. It uses **Quantum Espresso** (QE) as the underlying DFT engine.

## Functionality

The worker implements a **Delta Learning** strategy:
1.  Calculates the DFT energy and forces for a given structure.
2.  Calculates the baseline energy and forces using a Shifted Lennard-Jones potential.
3.  Output the difference (Delta) as the target label for the Machine Learning Potential.

$$ E_{target} = E_{DFT} - E_{LJ} $$

## Docker Image

The image is based on `quay.io/quantum-espresso/qe:7.3` and includes:
*   Python 3 + ASE
*   Quantum Espresso binaries (`pw.x`)
*   Shared project code

**Build Command:**
```bash
# From repository root
docker build -t dft_worker:latest -f workers/dft_worker/Dockerfile .
```

## Usage

The worker is typically invoked by the Orchestrator, but can be run manually for debugging.

**Command:**
```bash
docker run --rm -v $(pwd)/data:/data dft_worker:latest \
    python /app/src/main.py \
    --config /data/config.yaml \
    --meta-config /data/meta_config.yaml \
    --structure /data/input_structure.xyz \
    --output /data/labeled_output.xyz
```

### Arguments

*   `--config`: Path to the main experiment configuration file (inside container).
*   `--meta-config`: Path to the meta configuration file (inside container).
*   `--structure`: Path to the input structure file (XYZ format).
*   `--output`: Path where the labeled structure will be saved (XYZ format).

## Requirements

*   **Pseudopotentials**: The `config.yaml` must point to a valid directory containing Pseudopotentials (e.g., SSSP) accessible inside the container (usually mounted via `/data` or baked into the image).
