#!/bin/bash
set -e

# URLs for SSSP Efficiency 1.3.0 (PBE)
# From Materials Cloud: https://archive.materialscloud.org/records/rcyfm-68h65
SSSP_JSON_URL="https://archive.materialscloud.org/records/rcyfm-68h65/files/SSSP_1.3.0_PBE_efficiency.json?download=1"
SSSP_TAR_URL="https://archive.materialscloud.org/records/rcyfm-68h65/files/SSSP_1.3.0_PBE_efficiency.tar.gz?download=1"

# Target Directories
DATA_DIR="data"
PSEUDO_DIR="${DATA_DIR}/pseudos"

echo "Setting up SSSP Efficiency (PBE) Pseudopotentials..."

# Create directories
mkdir -p "$PSEUDO_DIR"

# Download JSON
echo "Downloading SSSP JSON..."
wget -O "${DATA_DIR}/sssp.json" "$SSSP_JSON_URL"

# Download and Extract Tarball
echo "Downloading and extracting SSSP Tarball..."
wget -O "${DATA_DIR}/sssp.tar.gz" "$SSSP_TAR_URL"

echo "Extracting to ${PSEUDO_DIR}..."
# The tarball usually contains a top-level directory. We strip it.
# If the tarball does not have a top-level directory, --strip-components=1 might fail or strip files.
# Let's inspect the tarball content first? No, we can't easily.
# Standard SSSP tarballs usually have a folder "SSSP_1.3.0_PBE_efficiency".
# We'll use tar's transform or just extract and move if needed.
# But --strip-components=1 is safer if we know it has a folder.
# Let's try --strip-components=1, if it fails, we handle it?
# To be robust, let's extract to a temporary directory.

TEMP_DIR=$(mktemp -d)
tar -xzf "${DATA_DIR}/sssp.tar.gz" -C "$TEMP_DIR"

# Move files from the extracted directory (whatever it is called) to target
# Usually there is one folder.
SUBDIR=$(ls "$TEMP_DIR" | head -n 1)
if [ -d "$TEMP_DIR/$SUBDIR" ]; then
    echo "Found subdirectory: $SUBDIR. Moving contents..."
    mv "$TEMP_DIR/$SUBDIR"/* "$PSEUDO_DIR/"
else
    echo "No subdirectory found. Moving files directly..."
    mv "$TEMP_DIR"/* "$PSEUDO_DIR/"
fi

# Cleanup
rm "${DATA_DIR}/sssp.tar.gz"
rm -rf "$TEMP_DIR"

echo "SSSP setup complete."
