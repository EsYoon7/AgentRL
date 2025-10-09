### Common dependencies for the training environment
# May not be up to date, double-check before using

FROM nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_BUILD_ISOLATION=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_ROOT_USER_ACTION=ignore
ARG PIP_INDEX_URL=https://pypi.org/simple

WORKDIR /workspace

### 1. install python
RUN apt-get update && \
    apt-get install -y python-is-python3 python3 python3-dev python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

### 2. python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade setuptools packaging pybind11
# note: pin torch version to the one requested by sglang before building
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cu129 torch==2.8.0 torchvision
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install sglang[all]==0.5.2 megatron-core transformer-engine[pytorch] flash-attn accelerate binpacking wandb ray[rllib] tensordict

### 3. configure utils
RUN apt-get update && \
    apt-get install -y parallel htop tmux && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install nvitop py-spy && \
    echo 'set-option -g history-limit 50000' > /root/.tmux.conf && \
    echo 'set-option -g mouse on' >> /root/.tmux.conf && \
    echo 'alias tt="tmux attach -t"' >> /root/.bashrc && \
    echo 'alias tn="tmux new -s"' >> /root/.bashrc && \
    echo 'alias dp="ls -A | parallel du -sh 2>/dev/null | sort -h"' >> /root/.bashrc && \
    echo 'alias ds="du -sh .[!.]* * 2>/dev/null | sort -h"' >> /root/.bashrc && \
    echo 'alias pd="py-spy dump --pid"' >> /root/.bashrc
