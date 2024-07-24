#!/bin/bash


MODEL_TYPE=phi-2
TARGET_DIR=bunny-phi-2

python -m bunny.eval.eval_spatialbench \
    --model-type $MODEL_TYPE\
    --data-path ./eval/spatial_bench \
    --conv-mode bunny \
    --question size.json \
    --depth
