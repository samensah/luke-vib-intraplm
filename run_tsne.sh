#!/bin/bash

python src/tsne.py \
    --checkpoints "outputs/lb_refind/-3/checkpoint-1500" \
                  "outputs/lb_refind/-2/checkpoint-1500" \
                  "outputs/lb_refind/0/checkpoint-1500" \
                  "outputs/lb_refind/6/checkpoint-1500" \
                  "outputs/lb_refind/-1/checkpoint-1500" \
    --data_file data/refind/test_cf.json \
    --device cuda


# python src/tsne.py \
#     --checkpoints "outputs/lb_refind/-3/checkpoint-1500" \
#                  "outputs/lb_refind/-2/checkpoint-1500" \
#                  "outputs/lb_refind/0/checkpoint-1500" \
#                  "outputs/lb_refind/1/checkpoint-1500" \
#                  "outputs/lb_refind/2/checkpoint-1500" \
#                  "outputs/lb_refind/3/checkpoint-1500" \
#                  "outputs/lb_refind/4/checkpoint-1500" \
#                  "outputs/lb_refind/5/checkpoint-1500" \
#                  "outputs/lb_refind/6/checkpoint-1500" \
#                  "outputs/lb_refind/7/checkpoint-1500" \
#                  "outputs/lb_refind/8/checkpoint-1500" \
#                  "outputs/lb_refind/9/checkpoint-1500" \
#                  "outputs/lb_refind/10/checkpoint-1500" \
#                  "outputs/lb_refind/-1/checkpoint-1500" \
#     --data_file data/refind/test.json \
#     --device cuda

# python src/tsne.py \
#     --checkpoints "outputs/lb_refind/-3/checkpoint-1500" \
#                  "outputs/lb_refind/-2/checkpoint-1500" \
#                  "outputs/lb_refind/0/checkpoint-1500" \
#                  "outputs/lb_refind/1/checkpoint-1500" \
#                  "outputs/lb_refind/2/checkpoint-1500" \
#                  "outputs/lb_refind/3/checkpoint-1500" \
#                  "outputs/lb_refind/4/checkpoint-1500" \
#                  "outputs/lb_refind/5/checkpoint-1500" \
#                  "outputs/lb_refind/6/checkpoint-1500" \
#                  "outputs/lb_refind/7/checkpoint-1500" \
#                  "outputs/lb_refind/8/checkpoint-1500" \
#                  "outputs/lb_refind/9/checkpoint-1500" \
#                  "outputs/lb_refind/10/checkpoint-1500" \
#                  "outputs/lb_refind/-1/checkpoint-1500" \
#     --data_file data/refind/test_cf.json \
#     --device cuda