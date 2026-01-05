#!/bin/bash

echo "Starting training ..."

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train TRAIN.yaml

echo "Training finished."