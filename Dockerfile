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

# Test code and report tooling live in the image. The dataset is mounted
# read-only by docker-compose so native source/reference files are never copied,
# resized, or modified during image construction.
COPY tests ./tests
COPY examples ./examples
COPY scripts ./scripts
RUN mkdir -p /app/results /app/dataset

ENTRYPOINT ["dark-channel-deblur"]
