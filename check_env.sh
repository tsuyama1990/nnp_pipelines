#!/bin/bash
set -e

echo "Checking environment dependencies..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Error: docker is not installed."
    exit 1
else
    echo "✅ docker is installed."
fi

# Check nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found. GPU features may not work."
else
    echo "✅ nvidia-smi is available."
fi

# Check Docker GPU access
echo "Checking Docker GPU access..."
if command -v docker &> /dev/null; then
    if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo "✅ Docker GPU access confirmed."
    else
        echo "⚠️  Warning: Docker cannot access GPUs. Ensure nvidia-container-toolkit is installed and configured."
    fi
else
    echo "Skipping Docker GPU check (docker not found)."
fi

# Check uv
if ! command -v uv &> /dev/null; then
    echo "⚠️  Warning: uv is not installed. Python dependency management might be slower."
else
    echo "✅ uv is installed."
fi

echo "Environment check passed!"
exit 0

# Check for LAMMPS PACE support (Optional/Warning)
echo "Checking LAMMPS PACE support..."
if command -v docker &> /dev/null; then
    # We check if the image exists first to avoid pulling/errors if not built
    if docker image inspect lammps_worker:latest &> /dev/null; then
        if docker run --rm lammps_worker:latest lmp_mpi -h | grep -q "pace"; then
            echo "✅ LAMMPS worker supports PACE."
        else
            echo "⚠️  Warning: lammps_worker image found but 'pace' style not detected in lmp_mpi -h."
            echo "    Ensure your LAMMPS image is built with the ML-PACE package."
        fi
    else
        echo "ℹ️  lammps_worker:latest image not found. Skipping PACE check (run 'docker-compose build' first)."
    fi
fi
