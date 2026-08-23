FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[dev]"

# Tests, reference material, dataset bootstrap and report tooling are part of
# the validation image. The test command downloads/extracts the authors'
# official image folder into /app/dataset/image before benchmarking it.
COPY tests ./tests
COPY examples ./examples
COPY dataset ./dataset
COPY scripts ./scripts
RUN mkdir -p /app/results /app/dataset/image

ENTRYPOINT ["dark-channel-deblur"]
