#!/bin/bash
set -e

# Script to download and unpack SSSP (Standard Solid State Pseudopotentials)
# for Quantum Espresso.

PSEUDO_DIR="./pseudos"
SSSP_URL="https://archive.materialscloud.org/record/file?filename=SSSP_precision_pseudos.tar.gz&record_id=1460"
SSSP_FILENAME="SSSP_precision_pseudos.tar.gz"

echo "Setting up SSSP Pseudopotentials..."

# Create directory if it doesn't exist
if [ ! -d "$PSEUDO_DIR" ]; then
    mkdir -p "$PSEUDO_DIR"
    echo "Created directory: $PSEUDO_DIR"
fi

cd "$PSEUDO_DIR"

# Download if not already present
if [ ! -f "$SSSP_FILENAME" ]; then
    echo "Downloading SSSP Precision Pseudos..."
    # Using curl with -L to follow redirects (Materials Cloud uses redirects)
    curl -L -o "$SSSP_FILENAME" "$SSSP_URL"
    echo "Download complete."
else
    echo "Archive $SSSP_FILENAME already exists. Skipping download."
fi

# Unpack
echo "Unpacking..."
tar -xzf "$SSSP_FILENAME"

# Optional: Clean up archive
# rm "$SSSP_FILENAME"

echo "✅ Pseudopotentials setup complete in $PSEUDO_DIR"
echo "   Please ensure 'config_meta.yaml' points to this directory:"
echo "   dft: { pseudo_dir: \"./pseudos\" }"
