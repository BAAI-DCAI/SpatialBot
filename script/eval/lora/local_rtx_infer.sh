#!/bin/bash

MODEL_TYPE=phi-2
MODEL_BASE=/path/to/base_llm_model
TARGET_DIR=bunny-lora-phi-2

python -m bunny.eval.eval_rtx \
    --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR\
    --model-base $MODEL_BASE\
    --model-type $MODEL_TYPE\
    --data-path /path/to/image/dir/\
    --conv-mode bunny \
    --question /path/to/open-loop-data-json