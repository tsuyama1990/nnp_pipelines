#!/bin/bash
set -e

PSEUDO_DIR="./pseudos"
SSSP_URL="https://archive.materialscloud.org/record/file?filename=SSSP_1.3.0_PBE_efficiency.tar.gz&record_id=1460"
TAR_NAME="SSSP_efficiency.tar.gz"

echo "Setting up Pseudopotentials..."
echo "Target Directory: $PSEUDO_DIR"

if [ -d "$PSEUDO_DIR" ]; then
    echo "Directory $PSEUDO_DIR already exists."
    # Optional: check if empty?
else
    mkdir -p "$PSEUDO_DIR"
fi

echo "Downloading SSSP Efficiency..."
if command -v wget &> /dev/null; then
    wget -O "$TAR_NAME" "$SSSP_URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$TAR_NAME" "$SSSP_URL"
else
    echo "Error: Neither wget nor curl found."
    exit 1
fi

echo "Extracting..."
tar -xf "$TAR_NAME" -C "$PSEUDO_DIR"

# Cleanup
rm "$TAR_NAME"

echo "✅ Pseudopotentials setup complete."
echo "Please update your config.yaml 'dft_params' to point to: $PSEUDO_DIR"
echo "Example:"
echo "dft_params:"
echo "  pseudo_dir: $PSEUDO_DIR"
