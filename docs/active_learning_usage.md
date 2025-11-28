# Active Learning with MD and KMC

This document explains how to configure and run Active Learning (AL) pipelines using different exploration strategies (MD, KMC, Hybrid) and `pyace` potentials.

## Overview

The Active Learning Orchestrator has been refactored to support pluggable exploration strategies. You can now easily switch between:

*   **LAMMPS MD (`lammps_md`)**: Standard Molecular Dynamics using LAMMPS.
*   **KMC (`kmc`)**: Kinetic Monte Carlo simulations.
*   **Hybrid (`hybrid`)**: A combined approach running MD followed by KMC (legacy behavior).
*   **ASE MD (`ase_md`)**: Simple MD using ASE (experimental).

## Configuration

The exploration strategy is controlled via the `exploration` section in `config.yaml`.

### Enable LAMMPS MD
To run pure MD with Active Learning checks:

```yaml
exploration:
  strategy: "lammps_md"

md_params:
  timestep: 1.0
  temperature: 300.0
  n_steps: 10000
  # ... other MD params
```

### Enable KMC
To run KMC exploration (requires a starting structure or restart file):

```yaml
exploration:
  strategy: "kmc"

kmc_params:
  active: true
  temperature: 300.0
  n_searches: 10
  # ... other KMC params
```

### Enable Hybrid (MD + KMC)
This is the default mode used in previous versions.

```yaml
exploration:
  strategy: "hybrid"
```

## Running with PyACE Potentials

The system is designed to work with ACE potentials (e.g., trained via `pacemaker`).

1.  **Initial Potential**: Ensure you have a `.yace` file (e.g., `potential.yace`).
2.  **Config**: Point to this potential in `config.yaml` under `al_params`.

```yaml
al_params:
  initial_potential: "path/to/potential.yace"
  potential_yaml_path: "path/to/potential.yaml" # Defines basis set
```

The `orchestrator` will automatically load this potential.
*   For **LAMMPS**, it uses the `pace` pair style.
*   For **KMC**, it uses the configured engine (likely LAMMPS or a Python wrapper around the potential).
*   For **ASE MD**, you may need to ensure a calculator compatible with the potential is available.

## Example Usage

See `examples/lammps_md_run_example.py` for a programmatic example of how to initialize the Orchestrator with a specific strategy.

To run the full pipeline:

```bash
python orchestrator/main.py --config config.yaml --meta meta_config.yaml
```

The system will:
1.  Load the configuration.
2.  Instantiate the selected `Explorer` (MD, KMC, etc.).
3.  Run the loop:
    *   **Explore**: Run MD/KMC.
    *   **Check Uncertainty**: If the potential extrapolation grade (gamma) exceeds `gamma_threshold`.
    *   **Active Learning**: If uncertain, select structures, label with DFT, and retrain the potential.
    *   **Resume**: Continue exploration with the updated potential.
