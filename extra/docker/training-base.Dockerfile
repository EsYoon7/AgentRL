### Common dependencies for the training environment
# May not be up to date, double-check before using

FROM nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-devel-ubuntu24.04

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
    uv pip install --system --upgrade setuptools packaging pybind11
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
      --extra-index-url https://download.pytorch.org/whl/cu128  \
      torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
RUN --mount=type=cache,target=/root/.cache/uv \
    echo "flashinfer-python==0.2.11.post3" > /tmp/overrides.txt && \
    uv pip install --system --override /tmp/overrides.txt \
      sglang[all]==0.4.8.post1 megatron-core transformer-engine[pytorch] \
      flash-attn accelerate binpacking wandb ray[rllib] tensordict nvitop py-spy && \
    rm -f /tmp/overrides.txt

### 3. configure utils
RUN echo 'set -g default-terminal "tmux-256color"' > /root/.tmux.conf && \
    echo "set -ga terminal-overrides ',*:Tc'" >> /root/.tmux.conf && \
    echo 'set-environment -g LANG "C.UTF-8"' >> /root/.tmux.conf && \
    echo 'set-environment -g LC_ALL "C.UTF-8"' >> /root/.tmux.conf && \
    echo 'set-option -g history-limit 50000' >> /root/.tmux.conf && \
    echo 'set-option -g mouse on' >> /root/.tmux.conf && \
    echo 'alias tt="tmux attach -t"' >> /root/.bashrc && \
    echo 'alias tn="tmux new -s"' >> /root/.bashrc && \
    echo 'alias dp="ls -A | parallel du -sh 2>/dev/null | sort -h"' >> /root/.bashrc && \
    echo 'alias ds="du -sh .[!.]* * 2>/dev/null | sort -h"' >> /root/.bashrc && \
    echo 'alias pd="py-spy dump --pid"' >> /root/.bashrc
