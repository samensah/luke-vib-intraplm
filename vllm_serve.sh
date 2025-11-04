#!/bin/bash
# Serve model with vLLM
export CUDA_VISIBLE_DEVICES=0

python -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-4-mini-instruct \
    --port 8000 \
    --host 0.0.0.0