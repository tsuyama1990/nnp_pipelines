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

# Check uv or pip
if command -v uv &> /dev/null; then
    echo "✅ uv is installed."
elif command -v pip &> /dev/null; then
    echo "✅ pip is installed."
else
    echo "⚠️  Warning: neither uv nor pip is found."
fi

echo "Environment check passed!"
