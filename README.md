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

## 🚀 Usage Workflow

本パイプラインは、再現性と設定管理のために `setup_experiment.py` を唯一のエントリーポイントとして設計されています。
直接 `orchestrator/main.py` を実行することは推奨されません。

### 1. Configuration (設定)
実験の設定は `config.yaml` で管理します。
目的に応じて設定ファイルをコピー・編集してください。

```bash
cp config.yaml my_experiment_config.yaml
# vim my_experiment_config.yaml
```

### 2\. Initialize & Run Experiment (実行)

`setup_experiment.py` を介して実験を開始します。このスクリプトは以下の処理を自動化します：

1.  **Workspace作成:** ユニークな実験IDを持つディレクトリ（`experiments/YYYYMMDD_HHMMSS_Name`）を作成。
2.  **Config凍結:** 使用した設定ファイルを実験ディレクトリ内にコピー（再現性の担保）。
3.  **初期化:** Seed生成、初期ポテンシャルの準備。
4.  **パイプライン起動:** `ActiveLearningOrchestrator` のプロセスを開始。

#### 基本コマンド

```bash
# デフォルト設定で実行
uv run setup_experiment.py

# 設定ファイルを指定して実行（推奨）
uv run setup_experiment.py --config my_experiment_config.yaml

# 実験名（タグ）を付けて実行
uv run setup_experiment.py --config config.yaml --name "al_ni_system_v1"
```

### 3\. Directory Structure (出力構造)

実行後、以下のディレクトリ構造が自動生成されます。

```text
work/
└── 07_active_learning/          # アクティブラーニングのメイン作業領域
    ├── experiment_state.json    # 中断再開用のステートファイル
    ├── config_snapshot.yaml     # 実行時の設定（凍結）
    ├── iteration_1/             # イテレーションごとの計算結果
    │   ├── candidate.xyz
    │   ├── train.xyz
    │   └── potential_v1.yace
    └── logs/
        └── experiment.log
```

### 4\. Resume / Restart (中断と再開)

実験が中断した場合、生成された実験ディレクトリを指定して再開します。

```bash
# 特定の実験ディレクトリから再開する場合
uv run setup_experiment.py --resume work/07_active_learning/ --iteration 5
```

## Directory Structure

*   `orchestrator/`: Python code for the control logic.
    *   `src/setup/`: Modules for experiment initialization.
    *   `src/wrappers/`: Docker wrappers that construct CLI commands for workers.
    *   `src/services/`: Business logic for MD, KMC, and Active Learning.
    *   `src/utils/`: Utility classes, including parallel execution helpers.
*   `workers/`: Source code and Dockerfiles for computational workers.
    *   `dft_worker/`: Quantum Espresso wrapper.
    *   `gen_worker/`: MACE structure generation and PyXtal integration.
    *   `pace_worker/`: Pacemaker training and sampling.
    *   `lammps_worker/`: LAMMPS MD/KMC engine.
*   `shared/`: Common Python code (Config, Data Structures) shared between Host and Workers.
*   `data/`: Runtime data directory (mounted to containers).

## License

[Insert License Here]
