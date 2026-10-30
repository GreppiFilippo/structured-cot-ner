FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL maintainer="pilener-cot-pipeline"

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace
ENV APP_PATH=/workspace

RUN apt-get update -y && \
    apt-get install -y curl \
                        git \
                        bash \
                        nano \
                        python3.11 \
                        python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed

# torch==2.6.0 is both the lab's recommended version for this cluster's
# drivers AND vllm==0.8.5's own exact pin (verified in its package
# metadata) — installed explicitly with the +cu124 build (CUDA 12.4, safely
# under the 12.5 ceiling confirmed via nvidia-smi) so the later general
# install resolves torch as already satisfied instead of pulling a
# default/newer build.
RUN pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

COPY build/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV DEBIAN_FRONTEND=dialog
