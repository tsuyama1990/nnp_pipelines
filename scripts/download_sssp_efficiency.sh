#!/bin/bash
set -e

# Download SSSP Efficiency Pseudopotentials
# This script downloads and extracts the SSSP Efficiency set to ./pseudos

PSEUDO_DIR="./pseudos"
SSSP_URL="https://archive.materialscloud.org/records/rcyfm-68h65/files/SSSP_1.3.0_PBE_efficiency.tar.gz?download=1"

if [ -d "$PSEUDO_DIR" ]; then
    echo "Directory $PSEUDO_DIR already exists."
else
    mkdir -p "$PSEUDO_DIR"
fi

echo "Downloading SSSP Efficiency from Materials Cloud..."
wget -O "$PSEUDO_DIR/sssp.tar.gz" "$SSSP_URL"

echo "Extracting..."
cd "$PSEUDO_DIR"
tar -xzvf sssp.tar.gz
rm sssp.tar.gz

echo "--------------------------------------------------------"
echo "Pseudopotentials have been downloaded to:"
echo "$(pwd)"
echo "Please update your config_meta.yaml to point to this directory."
echo "--------------------------------------------------------"
