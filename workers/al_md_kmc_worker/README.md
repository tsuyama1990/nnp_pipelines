# LAMMPS Worker

This worker runs Molecular Dynamics (MD) and Kinetic Monte Carlo (kMC) simulations using **LAMMPS**. It relies on the `ML-PACE` package to use the ACE potentials trained by the `pace_worker`.

## Functionality

1.  **MD**: Runs Molecular Dynamics simulations. It monitors the "extrapolation grade" (gamma) to detect uncertainty.
2.  **Small Cell**: Relaxes small unit cells or performs specific structural optimizations.
3.  **KMC**: Performs Off-Lattice KMC steps (saddle point search and migration).

## Docker Image

The image must be built with a custom LAMMPS version that includes `ML-PACE`.

**Build Command:**
```bash
# From repository root
docker build -t lammps_worker:latest -f workers/lammps_worker/Dockerfile .
```

## Usage

### 1. Molecular Dynamics (MD)

Runs an MD simulation until completion or until the uncertainty threshold is breached.

```bash
docker run --rm -v $(pwd)/data:/data lammps_worker:latest \
    python /app/src/main.py md \
    --config /data/config.yaml \
    --meta-config /data/meta_config.yaml \
    --potential /data/current_potential.yace \
    --structure /data/initial_structure.xyz \
    --steps 100000 \
    --gamma 2.0
```

**Arguments:**
*   `--potential`: Path to the ACE potential (.yace).
*   `--steps`: Number of MD steps.
*   `--gamma`: Uncertainty threshold (extrapolation grade).
*   `--restart`: (Optional) Flag to indicate restarting from a checkpoint.

### 2. Small Cell Relaxation

```bash
docker run --rm -v $(pwd)/data:/data lammps_worker:latest \
    python /app/src/main.py small_cell \
    --config /data/config.yaml \
    --meta-config /data/meta_config.yaml \
    --structure /data/cell.xyz \
    --center 0 \
    --potential /data/current_potential.yace \
    --output /data/relaxed_cell.xyz
```

### 3. Kinetic Monte Carlo (KMC)

```bash
docker run --rm -v $(pwd)/data:/data lammps_worker:latest \
    python /app/src/main.py kmc \
    --config /data/config.yaml \
    --meta-config /data/meta_config.yaml \
    --structure /data/current_state.xyz \
    --potential /data/current_potential.yace \
    --output /data/next_state.xyz
```
