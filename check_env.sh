#!/bin/bash
set -e

echo "Checking environment dependencies..."

if ! command -v docker &> /dev/null
then
    echo "❌ docker could not be found"
    exit 1
else
    echo "✅ docker is installed"
fi

if ! command -v nvidia-smi &> /dev/null
then
    echo "⚠️ nvidia-smi could not be found. GPU acceleration might not work."
else
    echo "✅ nvidia-smi is installed"
fi

if ! command -v uv &> /dev/null
then
    echo "❌ uv could not be found"
    exit 1
else
    echo "✅ uv is installed"
fi

echo "Environment check passed!"
