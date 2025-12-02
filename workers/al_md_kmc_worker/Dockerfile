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
#
# INSTRUCTIONS FOR PACE SUPPORT:
# 1. Clone LAMMPS source: git clone -b stable https://github.com/lammps/lammps.git
# 2. Clone PACE lib: git clone https://github.com/ICAMS/lammps-user-pace.git lammps/lib/pace
# 3. Build LAMMPS with cmake, enabling -D PKG_ML-PACE=ON.
# 4. Replace the base image 'lammps/lammps:stable' with your custom image.
#
# A future update will automate this build process. For now, we assume the user might provide a
# custom base image or accepts that 'pace' style won't work with the default image.
RUN pip3 install --no-cache-dir --break-system-packages ase numpy pyyaml lammps scipy

# Setup Shared Directory
COPY shared /app/shared
ENV PYTHONPATH=/app

# Copy Worker Source
COPY workers/al_md_kmc_worker/src /app/src

WORKDIR /data
