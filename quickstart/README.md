# NNP Pipeline Quickstart

This guide will help you set up and run a demo experiment (Aluminum-Copper) using the NNP Pipelines.

## Prerequisites

1.  **Docker** & **NVIDIA GPU** (nvidia-container-toolkit configured).
2.  **uv** or **pip** (for Python dependencies).

## 1. Setup Environment

Run the check script to ensure your environment is ready:

```bash
./check_env.sh
```

Build the worker containers:

```bash
make build
```

Start the background services:

```bash
make up
```

## 2. Validate Configuration

Validate the demo configuration:

```bash
uv run python validate_config.py quickstart/demo_config.yaml
```

## 3. Setup & Run Experiment

Initialize the experiment directory:

```bash
uv run python setup_experiment.py --config quickstart/demo_config.yaml --name AlCu_Demo
```

Run the pipeline:

```bash
./output/AlCu_Demo/run_pipeline.sh
```

## Monitoring

Logs will be streamed to the console. Artifacts are saved in `output/AlCu_Demo`.
