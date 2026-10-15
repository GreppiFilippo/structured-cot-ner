# Image of the free vs structured CoT experiment on the cluster (RTX 3090,
# NVIDIA driver 570 = CUDA <= 12.8): training (train_qwen.py) and evaluation
# (eval_tutor.py, vLLM) of Qwen3-4B-Thinking-2507 in one environment.
#
# Base: the official vLLM image (CUDA 12.8.1, Python 3.12, torch and vLLM
# built together), so no CUDA/torch/vLLM version has to be matched by hand.
# The Modal image (modal_app.py) uses vLLM 0.30 on CUDA 13, which this
# driver cannot run; Qwen3 models are supported by vLLM since 0.8.5.
#
# Build on the node that runs the jobs, from the repository root:
#   docker build -f 05_openvocab_sft/build_tutor/Dockerfile -t tutor-cot-image .
FROM vllm/vllm-openai:v0.10.1.1

# transformers >= 4.56 accepts from_pretrained(dtype=...), used by train_qwen.py
# (vLLM 0.10.1.1 requires >= 4.55)
RUN pip install --no-cache-dir "transformers==4.56.2" "peft==0.17.1" datasets sacremoses scipy

WORKDIR /workspace
# the base image starts the OpenAI-compatible server; here every job passes its own command
ENTRYPOINT []
