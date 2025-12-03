FROM ubuntu:22.04

USER root

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Quantum ESPRESSO and Python dependencies
# QE is available in Ubuntu repositories
RUN apt-get update && apt-get install -y \
    quantum-espresso \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
RUN pip3 install --no-cache-dir ase numpy pyyaml

# Setup Shared Directory
COPY shared /app/shared
ENV PYTHONPATH=/app

# Copy Worker Source
COPY workers/dft_worker/src /app/src

WORKDIR /data
