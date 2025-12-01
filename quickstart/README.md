# Quickstart Guide

This guide will help you get up and running with the NNP Active Learning Pipeline.

## Prerequisites

1.  **Docker**: Ensure Docker is installed and running.
2.  **NVIDIA GPU**: Required for `gen_worker` and `pace_worker` (or configure for CPU-only in `docker-compose.yml` if strictly needed, though GPU is recommended).
3.  **Python 3.10+**: For the orchestrator and setup scripts.

## Step 1: Environment Setup

Check your environment:
```bash
./check_env.sh
```

Build the worker images:
```bash
docker-compose build
docker-compose up -d
```
*Note: The containers will stay running in the background.*

## Step 2: Validate Configuration

We provide a demo configuration for a simple Al-Cu alloy system.

Validate it:
```bash
python3 validate_config.py quickstart/demo_config.yaml
```

## Step 3: Run the Experiment

Initialize the experiment directory:
```bash
python3 setup_experiment.py --config quickstart/demo_config.yaml --name demo_experiment
```

Run the pipeline:
```bash
# Option A: Run immediately (if you used --run in setup)
# Option B: Run the generated script
./output/demo_experiment/run_pipeline.sh
```

## Step 4: Monitor

Logs are available in `output/demo_experiment/logs/`.
Worker logs can be viewed via Docker:
```bash
docker logs -f gen_worker
```
