# Quickstart Guide

This guide will walk you through setting up and running a small demo experiment (Aluminum-Copper alloy) using ACE Active Carver.

## 1. Prerequisites

Ensure you have the following installed:
- Docker & Docker Compose
- NVIDIA Container Toolkit (for GPU acceleration)
- Python 3.8+ (for the orchestrator)

Verify your environment:
```bash
./check_env.sh
```

## 2. Build Containers

Build the worker images. This might take a few minutes.
```bash
docker-compose build
```
Start the worker services in the background:
```bash
docker-compose up -d
```

## 3. Prepare Configuration

We provide a demo configuration file `quickstart/demo_config.yaml` optimized for a quick test run.

Validate the configuration:
```bash
python3 validate_config.py quickstart/demo_config.yaml
```
*Expected Output: `✅ Configuration 'quickstart/demo_config.yaml' passed validation.`*

## 4. Setup Experiment

Initialize the experiment directory structure. This does **not** start the heavy calculations yet.
```bash
python3 setup_experiment.py --config quickstart/demo_config.yaml --name demo_run
```

You should see output indicating the directory `output/demo_run` was created.

## 5. Run the Pipeline

You can now start the active learning loop.

```bash
./output/demo_run/run_pipeline.sh
```

Or, if you want to run everything in one go from the start (step 4 + 5):
```bash
python3 setup_experiment.py --config quickstart/demo_config.yaml --name demo_run --run
```

## 6. Monitor Progress

Logs are written to the console and to `output/demo_run/logs/`.
You can check the status of workers using:
```bash
docker-compose ps
```

## Troubleshooting

- **GPU Errors:** If you see errors related to CUDA or GPUs, ensure `nvidia-smi` works on your host and inside the containers (check `check_env.sh` output).
- **Permissions:** Ensure your user has permission to read/write the `data/` directory.
