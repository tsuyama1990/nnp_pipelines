FROM tensorflow/tensorflow:2.15.0

# Install dependencies
# pacemaker is available on PyPI
RUN pip install --no-cache-dir pacemaker ase numpy pandas pyyaml scikit-learn numba scipy pyace pydantic

# Setup Shared Directory
COPY shared /app/shared
ENV PYTHONPATH=/app

# Copy Worker Source
COPY workers/pace_worker/src /app/src

WORKDIR /data
