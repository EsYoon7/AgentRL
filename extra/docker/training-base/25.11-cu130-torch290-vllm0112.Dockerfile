### Common dependencies for the training environment
# May not be up to date, double-check before using

FROM nvcr.io/nvidia/cuda-dl-base:25.11-cuda13.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy
ENV UV_NO_BUILD_ISOLATION=1
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

### 1. install python and base tooling
RUN apt-get update && \
    apt-get install -y \
      python-is-python3 python3 python3-dev \
      curl ca-certificates git htop ncurses-term parallel tmux && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

### 2. install uv and python dependencies
RUN curl -fsSL https://astral.sh/uv/install.sh | sh

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --upgrade setuptools packaging psutil ninja pybind11
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      https://github.com/vllm-project/vllm/releases/download/v0.11.2/vllm-0.11.2+cu130-cp38-abi3-manylinux1_x86_64.whl \
      https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.22/flash_attn-2.8.1+cu130torch2.9-cp312-cp312-linux_x86_64.whl \
      megatron-core transformer-engine[pytorch] \
      accelerate aiohttp binpacking filelock numpy Pillow \
      PyYAML ray[rllib] requests tensordict transformers \
      wandb nvitop py-spy

### 3. configure utils
RUN echo 'set -g default-terminal "tmux-256color"' > /root/.tmux.conf && \
    echo "set -ga terminal-overrides ',*:Tc'" >> /root/.tmux.conf && \
    echo 'set-environment -g LANG "C.UTF-8"' >> /root/.tmux.conf && \
    echo 'set-environment -g LC_ALL "C.UTF-8"' >> /root/.tmux.conf && \
    echo 'set-option -g history-limit 50000' >> /root/.tmux.conf && \
    echo 'set-option -g mouse on' >> /root/.tmux.conf && \
    echo 'alias pip="uv pip"' >> /root/.bashrc && \
    echo 'alias tt="tmux attach -t"' >> /root/.bashrc && \
    echo 'alias tn="tmux new -s"' >> /root/.bashrc && \
    echo 'alias dp="ls -A | parallel du -sh 2>/dev/null | sort -h"' >> /root/.bashrc && \
    echo 'alias ds="du -sh .[!.]* * 2>/dev/null | sort -h"' >> /root/.bashrc && \
    echo 'alias pd="py-spy dump --pid"' >> /root/.bashrc

### 4. install current agentrl trainer
COPY . /workspace/agentrl
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps \
      -e "/workspace/agentrl/trainer[vllm,megatron]"
