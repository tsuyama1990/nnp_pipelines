FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Install dependencies
RUN pip install --no-cache-dir mace-torch ase numpy pyyaml

# Setup Shared Directory
COPY shared /app/shared
ENV PYTHONPATH=/app

# Copy Worker Source
COPY workers/gen_worker/src /app/src

WORKDIR /data
