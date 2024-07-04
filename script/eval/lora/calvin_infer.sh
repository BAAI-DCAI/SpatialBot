#!/bin/bash

MODEL_TYPE=phi-2
MODEL_BASE=/path/to/base_llm_model
TARGET_DIR=bunny-lora-phi-2

python -m bunny.serve.calvin_infer \
    --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR\
    --model-base $MODEL_BASE\
    --model-type $MODEL_TYPE\
    --data-path /path/to/image/dir/\
    --image-path './ros_img/camera_image.png' \
    --answer-path './ros_answer/ros_answer.json' \
    --goal-path './ros_img/goal.json'