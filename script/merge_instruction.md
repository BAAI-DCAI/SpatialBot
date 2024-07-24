## To merge a LoRA tuned model: 
```
python script/merge_lora_weights.py \
  --model-path /path/to/bunny_lora_weights \
  --model-base /path/to/base_llm_model \
  --model-type phi-2 (or phi-3, llama3-8b, qwen1.5-0.5b, qwen1.5-1.8b) \
  --save-model-path /path/to/merged_model
```


##To linerally combine two models, model 1 and 2:

1. merge model 1, if it is LoRA tuned
```
python script/merge_lora_weights.py \
  --model-path ./checkpoints-llama3-8b/model1 \
  --model-base /path/to/meta-llama/Meta-Llama-3-8B-Instruct \
  --model-type llama3-8b \
  --save-model-path ./checkpoints-llama3-8b/model1-merged
```

2. merge model 2, if it is LoRA tuned
```
python script/merge_lora_weights.py \
  --model-path ./checkpoints-llama3-8b/model2 \
  --model-base /path/to/meta-llama/Meta-Llama-3-8B-Instruct \
  --model-type llama3-8b \
  --save-model-path ./checkpoints-llama3-8b/model2-merged
```

3. Copy configuration from model2
```
cp -r ./checkpoints-llama3-8b/model2-merged ./checkpoints-llama3-8b/model-avg
```

4. Linerally average weights of two models
```
from safetensors.torch import load_file, save_file

total = 4
for i in range(1, total + 1):
    model_1 = load_file(f'./checkpoints-llama3-8b/model1-merged/model-{i:05d}-of-{total:05d}.safetensors')
    model_2 = load_file(f'./checkpoints-llama3-8b/model1-merged/model-{i:05d}-of-{total:05d}.safetensors')
    
    assert model_1.keys() == model_2.keys()

    avg = {}
    for k in model_1.keys():
        avg[k] = model_1[k] * 0.4 + model_2[k] * 0.6 # the weight factor is selected empirically

    save_file(avg, f'./checkpoints-llama3-8b/bunny-llama3-8b-avg/model-{i:05d}-of-{total:05d}.safetensors', {'format': 'pt'})
```