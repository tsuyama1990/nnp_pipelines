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
# Note: The base image binary might NOT have PACE package. For production, a custom build is needed.
RUN pip3 install --no-cache-dir --break-system-packages ase numpy pyyaml lammps scipy

# Setup Shared Directory
COPY shared /app/shared
ENV PYTHONPATH=/app

# Copy Worker Source
COPY workers/lammps_worker/src /app/src

WORKDIR /data
