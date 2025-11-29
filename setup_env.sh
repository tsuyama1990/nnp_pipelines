#!/bin/bash
set -e

echo "=============================================="
echo "Setting up environment for ace-active-carver"
echo "=============================================="

# 1. Install uv if missing
if ! command -v uv &> /dev/null; then
    echo "'uv' is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env || export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Pin Python Version & Sync Environment
echo "Initializing Python environment..."

# FORCE Python 3.12 to avoid PyTorch/CPython 3.14 compatibility issues
uv python pin 3.12

echo "Installing dependencies..."
uv sync

echo "Verifying PyTorch and CUDA installation..."
uv run python check_gpu.py

echo "Creating directory structure..."
mkdir -p src tests data

# ==========================================================
# AUTOMATED LAMMPS INSTALLATION SECTION
# ==========================================================
echo ""
echo "=============================================="
echo "Automating LAMMPS Installation (ML-PACE)"
echo "=============================================="

# Check if lammps already exists to avoid re-cloning
if [ -d "mylammps" ]; then
    echo "Directory 'mylammps' already exists. Skipping clone/build."
    echo "Delete 'mylammps' folder if you want to rebuild."
else
    echo "1. Cloning LAMMPS repository..."
    git clone -b stable https://github.com/lammps/lammps.git mylammps

    echo "2. Configuring build with CMake..."
    mkdir -p mylammps/build
    cd mylammps/build

    # Get the python executable path from uv (now guaranteed to be 3.12)
    PYTHON_EXE=$(uv python find)
    echo "   Using Python: $PYTHON_EXE"

    # Configure CMake with PACE and Python packages enabled
    cmake ../cmake \
        -D PKG_PYTHON=ON \
        -D PKG_ML-PACE=ON \
        -D BUILD_SHARED_LIBS=ON \
        -D PYTHON_EXECUTABLE="$PYTHON_EXE" \
        -D CMAKE_INSTALL_PREFIX=$(pwd)/../install \
        -D PKG_KOKKOS=ON \
        -D Kokkos_ENABLE_SERIAL=ON

    # NOTE: If you have a GPU, add "-D Kokkos_ENABLE_CUDA=ON" above.

    echo "3. Building LAMMPS (This may take 10-20 minutes)..."
    make -j$(nproc)
    make install
    
    # Return to project root
    cd ../..
    
    echo "LAMMPS installed successfully in ./mylammps/install"
fi

# ==========================================================
# POST-INSTALLATION SETUP
# ==========================================================
# We need to ensure the new LAMMPS binary and library are found
export PATH="$(pwd)/mylammps/install/bin:$PATH"
export LD_LIBRARY_PATH="$(pwd)/mylammps/install/lib:$LD_LIBRARY_PATH"

# Dynamically find the python version (3.12) to construct the path
PY_VER_SHORT=$(uv run python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH="$(pwd)/mylammps/install/lib/python${PY_VER_SHORT}/site-packages:$PYTHONPATH"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "Python pinned to 3.12."
echo "LAMMPS has been compiled and added to your path."
echo "=============================================="