FROM python:3.11-slim

# Install system dependencies for building LAMMPS
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libopenmpi-dev \
    openmpi-bin \
    libfftw3-dev \
    libjpeg-dev \
    libpng-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV LAMMPS_VERSION=stable_2Aug2023_update3
ENV LAMMPS_DIR=/opt/lammps
ENV PYTHONPATH=/app:${LAMMPS_DIR}/python

# Clone LAMMPS source
WORKDIR /opt
RUN git clone --depth 1 --branch ${LAMMPS_VERSION} https://github.com/lammps/lammps.git ${LAMMPS_DIR}

# Build LAMMPS with ML-PACE package enabled
WORKDIR ${LAMMPS_DIR}/build
RUN cmake ../cmake \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=ON \
    -D PKG_ML-PACE=ON \
    -D PKG_PYTHON=ON \
    -D PKG_MOLECULE=ON \
    -D PKG_MANYBODY=ON \
    -D PKG_EXTRA-PAIR=ON \
    && cmake --build . -j$(nproc) \
    && cmake --install .

# Install Python dependencies
RUN pip3 install --no-cache-dir ase numpy pyyaml scipy matplotlib pandas

# Install pyace from GitHub (ICAMS/python-ace)
# This provides PyACECalculator and other ACE utilities
RUN pip3 install --no-cache-dir git+https://github.com/ICAMS/python-ace.git

# Verify installations
RUN python3 -c "import lammps; print('LAMMPS imported successfully')" && \
    python3 -c "from pyace import PyACECalculator; print('PyACECalculator imported successfully')" && \
    lmp -help || echo "LAMMPS binary check"

# Setup Shared Directory
COPY shared /app/shared
ENV PYTHONPATH=/app

# Copy Worker Source
COPY workers/al_md_kmc_worker/src /app/src

WORKDIR /data

# Note: This Dockerfile builds LAMMPS from source with ML-PACE support
# Build time: ~10-15 minutes depending on system
# Image size: ~2-3 GB (larger than pip version but includes full LAMMPS+PACE)
