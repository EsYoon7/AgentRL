FROM debian:12-slim AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV PYTHONDONTWRITEBYTECODE=1
ENV VENV_PATH=/venv

RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes \
      python3-venv gcc libpython3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VENV_PATH" && \
    "$VENV_PATH"/bin/pip install --upgrade pip setuptools wheel

ENV PATH="$VENV_PATH/bin:$PATH"

FROM base AS deps

COPY ./eval/pyproject.toml /tmp/pyproject.toml

RUN set -eux; \
    python3 - <<'PY' >/tmp/requirements.txt
import pathlib
import tomllib

data = tomllib.loads(pathlib.Path('/tmp/pyproject.toml').read_text())
deps = list(data.get('project', {}).get('dependencies', ()))
extras = data.get('project', {}).get('optional-dependencies', {})
deps.extend(extras.get('tui', ()))
print('\n'.join(deps))
PY

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --disable-pip-version-check --requirement /tmp/requirements.txt

FROM deps AS build-venv

COPY ./eval /app/eval

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --disable-pip-version-check --no-deps "/app/eval[tui]"

FROM gcr.io/distroless/python3-debian12:nonroot

COPY --from=build-venv /venv /venv

ENTRYPOINT ["/venv/bin/agentrl-eval"]
