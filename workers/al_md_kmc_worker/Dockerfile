FROM lammps/lammps:stable

USER root

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
# We install 'lammps' python package to get bindings if needed, but the binary comes from the base image.
#
# IMPORTANT: The base 'lammps/lammps:stable' image likely does NOT include the ML-PACE package.
# To use ACE potentials in LAMMPS, you must build a custom image with the PACE package enabled.
# Refer to the LAMMPS documentation on building with the ML-PACE package.
# A future update will automate this build process.
RUN pip3 install --no-cache-dir --break-system-packages ase numpy pyyaml lammps scipy

# Setup Shared Directory
COPY shared /app/shared
ENV PYTHONPATH=/app

# Copy Worker Source
COPY workers/al_md_kmc_worker/src /app/src

WORKDIR /data
