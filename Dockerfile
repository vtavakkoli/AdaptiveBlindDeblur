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

# CI validates every source image committed under dataset/image. The large
# historical result folders are excluded via .dockerignore so image builds stay
# compact while the real 23-image source set remains inside the container.
COPY tests ./tests
COPY examples ./examples
COPY dataset/image ./dataset/image
COPY scripts ./scripts
RUN mkdir -p /app/results

ENTRYPOINT ["dark-channel-deblur"]
