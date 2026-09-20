FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL maintainer="cot-sft-training"

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

# Same +cu124 pin used across this project's other Dockerfiles — matches
# the cluster's CUDA 12.5 driver ceiling (RTX 3090s on faretra/moro232).
# torchvision pinned to match torch==2.6.0 (same reasoning as
# bioner_eval/build/Dockerfile — MedGemma is multimodal, needs a working
# image processor backend even for text-only use).
RUN pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

COPY build/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV DEBIAN_FRONTEND=dialog
