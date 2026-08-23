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

# Regression assets and test/report tooling are intentionally copied into the
# image: docker-compose's test service validates the real reference images and
# writes a portable comparison report to /app/results.
COPY tests ./tests
COPY examples ./examples
COPY scripts ./scripts
RUN mkdir -p /app/results

ENTRYPOINT ["dark-channel-deblur"]
