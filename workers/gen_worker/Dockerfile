FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Install dependencies
# PyXtal requires scipy, spglib, pymatgen (which requires numpy, pandas, etc.)
RUN pip install --no-cache-dir mace-torch ase numpy pyyaml pyxtal

# Setup Shared Directory
COPY shared /app/shared
ENV PYTHONPATH=/app

# Copy Worker Source
COPY workers/gen_worker/src /app/src

WORKDIR /data
