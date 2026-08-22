FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build libfftw3-dev libpng-dev libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DDEBLUR_BUILD_TESTS=ON \
      -DDEBLUR_NATIVE_ARCH=OFF \
    && cmake --build build --parallel \
    && ctest --test-dir build --output-on-failure

ENTRYPOINT ["/app/build/deblur"]
CMD ["--help"]
