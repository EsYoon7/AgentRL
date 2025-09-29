FROM --platform=$BUILDPLATFORM node:lts AS dashboard-builder

WORKDIR /app

COPY controller/dashboard /app

RUN yarn install --frozen-lockfile

RUN yarn build

FROM --platform=$BUILDPLATFORM golang:latest AS builder

ARG BUILDOS
ARG BUILDARCH
ARG TARGETOS
ARG TARGETARCH
ARG TARGETVARIANT

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip && \
    rm -rf /var/lib/apt/lists/*

ARG PROTOC_VERSION=31.1

RUN set -eux; \
    arch="$BUILDARCH"; \
    case "$arch" in \
      amd64)  arch="x86_64" ;; \
      arm64)  arch="aarch_64" ;; \
      *) echo "unsupported arch: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    curl -L -o /tmp/protoc.zip \
      https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-${BUILDOS}-${arch}.zip && \
    unzip -o /tmp/protoc.zip -d /usr/local && \
    rm /tmp/protoc.zip && \
    go install google.golang.org/protobuf/cmd/protoc-gen-go@latest && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

COPY controller/go.mod controller/go.sum /app/
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    go mod download

COPY proto /tmp/proto
RUN mkdir -p /app/internal/pb && \
    find /tmp/proto -type f -name '*.proto' -print0 | \
    xargs -0 \
      protoc --proto_path=/tmp/proto \
      --go_out=internal/pb --go_opt=paths=source_relative \
      --go-grpc_out=internal/pb --go-grpc_opt=paths=source_relative

COPY controller /app
COPY --from=dashboard-builder /app/dist /app/dashboard/dist

RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    export CGO_ENABLED=0; \
    export GOOS="$TARGETOS"; \
    export GOARCH="$TARGETARCH"; \
    if [ "$TARGETARCH" = "arm" ] && [ -n "$TARGETVARIANT" ]; then \
      export GOARM="${TARGETVARIANT#v}"; \
    fi; \
    go build -a -o agentrl .

# check if the binary runs
RUN if [ "$TARGETOS" = "linux" ] && [ "$TARGETARCH" = "$(go env GOARCH)" ]; then \
      ./agentrl --help; \
    fi

FROM --platform=$BUILDPLATFORM scratch AS binary

COPY --from=builder /app/agentrl /agentrl

FROM gcr.io/distroless/base-debian12:nonroot

COPY --from=builder /app/agentrl /agentrl

EXPOSE 5020

ENTRYPOINT ["/agentrl"]
