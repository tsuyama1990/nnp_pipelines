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

# Check uv
if ! command -v uv &> /dev/null; then
    echo "⚠️  Warning: uv is not installed. Python dependency management might be slower."
else
    echo "✅ uv is installed."
fi

echo "Environment check passed!"
exit 0
