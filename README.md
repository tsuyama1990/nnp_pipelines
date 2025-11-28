# ACE Active Carver

An automated Active Learning system for materials science, designed to train Atomic Cluster Expansion (ACE) potentials using a Hybrid MD-kMC workflow. The system autonomously explores phase space, identifies high-uncertainty configurations, and retrains the potential using First-Principles (DFT) data.

## Architecture

This project follows a **Micro-kernel Architecture**:

*   **Orchestrator (Host)**: A lightweight Python application that manages the active learning loop, state, and decision logic. It does not perform heavy computations itself.
*   **Workers (Docker Containers)**: Specialized, isolated environments for heavy computational tasks. The Orchestrator invokes these workers via `docker run`.
    *   `gen_worker`: Generates candidate structures using MACE (Foundational ML Force Field) and PyXtal (Symmetry-based generation).
    *   `dft_worker`: Performs First-Principles calculations (Quantum Espresso) to label data.
    *   `pace_worker`: Trains ACE potentials and performs uncertainty-based sampling (Pacemaker).
    *   `lammps_worker`: Runs Molecular Dynamics (MD) and Kinetic Monte Carlo (kMC) simulations (LAMMPS).
*   **Shared Data**: Data is exchanged via a shared volume mounted at `./data` on the host and `/data` inside containers.

## Prerequisites

*   **Linux OS** (Ubuntu/Debian recommended)
*   **Docker Engine** (with non-root user access configured)
*   **NVIDIA Drivers** & **NVIDIA Container Toolkit** (required for GPU acceleration in `pace_worker` and `gen_worker`)
*   **uv** (Python package manager)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/ace-active-carver.git
    cd ace-active-carver
    ```

2.  **Install dependencies using `uv`:**
    ```bash
    # Install uv if not present
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Sync dependencies
    uv sync
    ```

3.  **Build Worker Images:**
    Each worker has its own `Dockerfile` in `workers/<name>`. You must build them before running the orchestrator.
    ```bash
    # Example build script usage (if available) or manual build:
    docker build -t dft_worker:latest -f workers/dft_worker/Dockerfile .
    docker build -t gen_worker:latest -f workers/gen_worker/Dockerfile .
    docker build -t pace_worker:latest -f workers/pace_worker/Dockerfile .
    docker build -t lammps_worker:latest -f workers/lammps_worker/Dockerfile .
    ```

## Usage

1.  **Configuration:**
    Edit `config.yaml` to set your experiment parameters (elements, temperature, DFT settings, etc.).

    *   **Structure Generation:** You can configure `gen_worker` to use strategies like `random_symmetry` (PyXtal) to explore new structures. See `workers/gen_worker/README.md` for details.

2.  **Run the Orchestrator:**
    ```bash
    uv run orchestrator/main.py
    ```

    The system will:
    1.  Check for an initial potential. If missing, it triggers `gen_worker` to create seed data.
    2.  Label data using `dft_worker`.
    3.  Train a potential using `pace_worker`.
    4.  Run MD/KMC simulations using `lammps_worker`.
    5.  Monitor uncertainty and loop back to step 2 if necessary.

## Directory Structure

*   `orchestrator/`: Python code for the control logic.
    *   `src/wrappers/`: Docker wrappers that construct CLI commands for workers.
    *   `src/services/`: Business logic for MD, KMC, and Active Learning.
*   `workers/`: Source code and Dockerfiles for computational workers.
    *   `dft_worker/`: Quantum Espresso wrapper.
    *   `gen_worker/`: MACE structure generation and PyXtal integration.
    *   `pace_worker/`: Pacemaker training and sampling.
    *   `lammps_worker/`: LAMMPS MD/KMC engine.
*   `shared/`: Common Python code (Config, Data Structures) shared between Host and Workers.
*   `data/`: Runtime data directory (mounted to containers).

## License

[Insert License Here]
