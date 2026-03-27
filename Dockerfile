# Production image: Rust four_d_engine (cdylib) + Go CLI/API with CGO.
FROM rust:1-bookworm AS rust
WORKDIR /src
COPY 4d-engine/Cargo.toml 4d-engine/Cargo.lock ./4d-engine/
COPY 4d-engine/src ./4d-engine/src
COPY 4d-engine/include ./4d-engine/include
RUN cd 4d-engine && cargo build --release

FROM golang:1.22-bookworm AS go
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal
COPY 4d-engine/include ./4d-engine/include
COPY --from=rust /src/4d-engine/target/release/libfour_d_engine.so /src/4d-engine/target/release/libfour_d_engine.so
ENV CGO_ENABLED=1
ENV LD_LIBRARY_PATH=/src/4d-engine/target/release:/usr/local/lib
RUN go build -o /out/4dollama ./cmd/4dollama

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --home /nonexistent --shell /usr/sbin/nologin fourd
COPY --from=rust /src/4d-engine/target/release/libfour_d_engine.so /usr/local/lib/libfour_d_engine.so
COPY --from=go /out/4dollama /usr/local/bin/4dollama
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV FOURD_GPU=cpu
ENV FOURD_HOST=0.0.0.0
ENV FOURD_PORT=13377
ENV FOURD_MODELS=/models
EXPOSE 13377
USER fourd
VOLUME ["/models"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1:13377/healthz || exit 1
ENTRYPOINT ["/usr/local/bin/4dollama"]
CMD ["serve"]
