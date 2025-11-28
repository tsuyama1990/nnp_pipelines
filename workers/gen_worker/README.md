# Generation Worker

This worker handles structure generation and filtering using **MACE** (Massively Atomic Chemical Expansion), a foundational Machine Learning Force Field, and **PyXtal** for random symmetry structure generation. It is primarily used during the "Seed Generation" phase to create an initial diverse dataset.

## Functionality

1.  **Generate**: Creates candidate structures using various scenarios (e.g., Random Symmetry Search, Defects, Surfaces) and can relax them using a pre-trained MACE model to ensure they are physically reasonable.
2.  **Filter**: Filters structures based on atomic distance checks and MACE energy/force criteria.

## Docker Image

**Build Command:**
```bash
# From repository root
docker build -t gen_worker:latest -f workers/gen_worker/Dockerfile .
```

*Note: This worker requires GPU support (NVIDIA Drivers) for MACE inference. PyXtal generation runs on CPU.*

## Usage

### 1. Generation

Generates structures based on configuration settings defined in `config.yaml`.

```bash
docker run --rm --gpus all -v $(pwd)/data:/data gen_worker:latest \
    python /app/src/main.py generate \
    --config /data/config.yaml \
    --output /data/generated_structures.xyz
```

**Arguments:**
*   `--config`: Path to config file.
*   `--output`: Output path for generated XYZ file.

### Configuration Scenarios

The worker supports multiple generation scenarios defined in the `generation.scenarios` list in `config.yaml`.

#### Random Symmetry Generation (PyXtal)

Uses `PyXtal` to generate random crystal structures with specific symmetry constraints (space groups) and optional composition control.

```yaml
generation:
  scenarios:
    - type: "random_symmetry"
      # Elements to include in the system
      elements: ["Al", "Cu"]
      # Number of structures to generate
      num_structures: 10
      # Range of Space Group numbers to explore (1-230)
      space_group_range: [1, 230]
      # Volume scaling factor
      volume_factor: 1.0
      # (Optional) Explicit composition constraint
      composition:
        Al: 4
        Cu: 4
      # Maximum attempts per structure to satisfy symmetry/distance constraints
      max_attempts: 50
```

### 2. Filtering

Filters an input XYZ file using checks for minimum atomic distance and MACE forces (e.g., removing high-force structures).

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
