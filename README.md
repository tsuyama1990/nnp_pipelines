# ACE Active Carver

 **ACE Active Carver** is a modular, Dockerized Active Learning (AL) pipeline for constructing Machine Learning Interatomic Potentials (MLIPs) using the Atomic Cluster Expansion (ACE) framework. It integrates **MACE** (for foundation model-based structure generation), **Pacemaker** (for ACE training), **LAMMPS** (for MD/KMC exploration), and **Quantum ESPRESSO** (for DFT labeling).

 ---

 ## 🏗 Architecture

 The project follows a **Micro-kernel Architecture**:
 *   **Orchestrator:** A lightweight Python core (in `workers/al_md_kmc_worker`) that manages the AL loop, state, and task delegation.
 *   **Workers:** Docker containers specialized for heavy computational tasks.
     *   `gen_worker`: Structure generation using PyXtal and MACE-MP relaxation.
     *   `lammps_worker`: Molecular Dynamics (MD) and Kinetic Monte Carlo (KMC) simulations.
     *   `dft_worker`: DFT calculations (Quantum ESPRESSO) for labeling.
     *   `pace_worker`: Active Learning sampling (MaxVol) and Potential Training (Pacemaker).

 ---

 ## 🚀 Getting Started

 ### Prerequisites

 Ensure you have the following installed:
 *   [Docker](https://docs.docker.com/get-docker/)
 *   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)
 *   [uv](https://github.com/astral-sh/uv) (for fast Python dependency management)

 You can verify your environment using the provided script:

 ```bash
 ./check_env.sh
 ```

 ### Installation

 1.  Clone the repository:
     ```bash
     git clone https://github.com/your-org/ace-active-carver.git
     cd ace-active-carver
     ```

 2.  Build the Docker images:
     ```bash
     make build
     ```

 ### Running an Experiment

 1.  **Configure:** Edit `config.yaml` to define your system (elements, temperature, etc.).

 2.  **Start:** Run the setup script.
     ```bash
     python setup_experiment.py --config config.yaml --name my_experiment
     ```
     This creates an experiment directory structure in `experiment_my_experiment/`.

 3.  **Execute:**
     The setup script generates a `run_pipeline.sh` inside `experiment_my_experiment/configs/`. You can uncomment the relevant steps in that script and run it, or rely on the orchestration if enabled.

 ### Development & Testing

 The project includes a comprehensive test suite.

 *   **Run Unit Tests:**
     ```bash
     make test
     ```
 *   **Run Integration Tests (Mocked Docker):**
     ```bash
     make test-integration
     ```
 *   **Clean Environment:**
     ```bash
     make clean
     ```

 ---

 ## 📂 Directory Structure

 ```text
 .
 ├── config.yaml             # Main experiment configuration
 ├── config_meta.yaml        # Environment-specific settings (Docker tags, commands)
 ├── setup_experiment.py     # Entry point script
 ├── docker-compose.yml      # Orchestration definition
 ├── Makefile                # Build/Run shortcuts
 ├── shared/                 # Common Python code (Config, Utils, Potentials)
 ├── workers/                # Source code for micro-services
 │   ├── al_md_kmc_worker/   # Orchestrator & LAMMPS
 │   ├── dft_worker/         # DFT Labeling
 │   ├── gen_worker/         # Structure Generation
 │   └── pace_worker/        # Training & Sampling
 └── tests/                  # Unit and Integration tests
 ```

 ## ⚖️ License

 This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
