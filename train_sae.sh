#!/bin/bash


CUDA_VISIBLE_DEVICES=1 python train_sae.py \
      --dictionary-size 8567 \
      --data-path /home/D/mj/data/vision_transfer.jsonl
