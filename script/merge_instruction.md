python script/merge_lora_weights.py \
  --model-path /path/to/bunny_lora_weights \
  --model-base /path/to/base_llm_model \
  --model-type phi-2 (or phi-3, llama3-8b, qwen1.5-0.5b, qwen1.5-1.8b) \
  --save-model-path /path/to/merged_model