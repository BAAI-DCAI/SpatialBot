<h1 align = "center">
  SpatialBot
</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2406.13642">
        <img alt="Paper" src="http://img.shields.io/badge/Paper-arXiv%3A2406.13642-B31B1B.svg">
    </a>
    <a href="https://huggingface.co/RussRobin/SpatialBot-3B">
        <img alt="Model SpatialBot-3B" src="https://img.shields.io/badge/🤗%20Model-SpatialBot--3B-green">
    </a>
    <a href="https://huggingface.co/datasets/RussRobin/SpatialBench">
        <img alt="Benchmark" src="https://img.shields.io/badge/🤗%20Benchmark-SpatialBench-blue">
    </a>
    <a href="https://mp.weixin.qq.com/s/X1iqkkEMsop9DGCY08AfCw">
        <img alt="News in Chinese" src="https://img.shields.io/badge/📰%20News_in_Chinese-机器之心-purple">
    </a>
</p>

[//]: # (<a href="https://huggingface.co/datasets/RussRobin/SpatialQA">)

[//]: # (        <img alt="Dataset" src="https://img.shields.io/badge/🤗%20Dataset-SpatialQA-yellow">)

[//]: # (    </a>)

This is the official repo for "SpatialBot: Precise Spatial Understanding with Vision Language Models".

SJTU, Stanford, BAAI, PKU, Oxford, SEU

<!-- ![comparison_8B](comparison_8B.png) -->

Model: [🤗3B model in HF](https://huggingface.co/RussRobin/SpatialBot-3B) | [🤗3B ckpt in HF](https://huggingface.co/RussRobin/SpatialBot-3B-LoRA) | [🤖3B model in wisemodel](https://wisemodel.cn/models/RussellRobin/SpatialBot-3B) | [🤖3B ckpt in wisemodel](https://wisemodel.cn/models/RussellRobin/SpatialBot-3B-LoRA)

Benchmark: [🤗SpatialBench in HF](https://huggingface.co/datasets/RussRobin/SpatialBench) | [🤖SpatialBench in wisemodel](https://wisemodel.cn/datasets/RussellRobin/SpatialBench/file)

Paper: [📃General VQA + embodiment arXiv](https://arxiv.org/abs/2406.13642)

Embodiment Videos Preview: [⚙ SpatialBot in embodiment](https://drive.google.com/drive/folders/1WBt5M0h2Z8k_ohPVUVEwIcCDaLxF9yv9?usp=sharing)
## 🚀 Quickstart

1. Install dependencies first:

```
pip install torch transformers accelerate pillow numpy
```

2. Download [SpatialBot-3B](https://huggingface.co/RussRobin/SpatialBot-3B). 
Users in mainland China may want to download HF model from [HF mirror site](https://hf-mirror.com/) and change ```model_name``` to local path of SpatialBot-3B folder.

3. Run the model:
```
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import numpy as np

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set device
device = 'cuda'  # or cpu

model_name = 'RussRobin/SpatialBot-3B'
offset_bos = 0

# create model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True)

# text prompt
prompt = 'What is the depth value of point <0.5,0.2>? Answer directly from depth map.'
text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:], dtype=torch.long).unsqueeze(0).to(device)

image1 = Image.open('rgb.jpg')
image2 = Image.open('depth.png')

channels = len(image2.getbands())
if channels == 1:
    img = np.array(image2)
    height, width = img.shape
    three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
    three_channel_array[:, :, 0] = (img // 1024) * 4
    three_channel_array[:, :, 1] = (img // 32) * 8
    three_channel_array[:, :, 2] = (img % 32) * 8
    image2 = Image.fromarray(three_channel_array, 'RGB')

image_tensor = model.process_images([image1,image2], model.config).to(dtype=model.dtype, device=device)

# If 'Expected all tensors to be on the same device' error is thrown, uncomment the following line
# model.get_vision_tower().to('cuda')

# generate
output_ids = model.generate(
    input_ids,
    images=image_tensor,
    max_new_tokens=100,
    use_cache=True,
    repetition_penalty=1.0 # increase this to avoid chattering
)[0]

print(tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip())
```

## 🤖 SpatialBot Installation
SpatialBot is a multi-image version of [Bunny](https://github.com/BAAI-DCAI/Bunny). 
If you've installed Bunny, just replace the code with ours are reinstall ```bunny``` package.
You can start from a docker or configure local environments.

### Start from Docker
We provide a ready to run [docker container](https://hub.docker.com/r/russellrobin/bunny). Please update it with our codes:
```
# 1. download docker image
docker pull russellrobin/bunny:latest

# 2. run container
# docker run -itd ...
# docker exec -it ...

# 3. upgrade transformers and bunny package
cd SpatialBot && pip install --upgrade transformers && pip uninstall bunny && pip install -e .
```

### Local Installation
Please follow the [instructions](https://github.com/BAAI-DCAI/Bunny?tab=readme-ov-file#local-installation), but use codes in this repo.

## 🏋 SpatialBot Training
Please [download](https://github.com/BAAI-DCAI/Bunny?tab=readme-ov-file#support-models) the base LLM and vision tower weights first.
To pretrain the model:
```
sh script/train/pretrain.sh
```
To finetune SpatialBot with LoRA:
```
sh script/train/finetune_lora.sh
```

Parameters:

```MODEL_TYPE```: base LLM type, we support ```phi-2, phi-3,qwen1.5-0.5b, qwen1.5-1.8b (4B), and llama3-8b```.

```PRETRAIN_DIR```: path to a pretrained model.

```OUTPUT_DIR```: path to save model.

```--model_name_or_path```: path to base LLM. 

```--vision_tower```: path to vision encoder. We support CLIP, SigLIP, and EVA-CLIP.

```--version```: for Phi-2 and QWen, use ```bunny```. For ```Phi-3/Llama3```, please use ```phi3/llama```

Please find finetuned ckpts of [SpatialBot-3B-LoRA](https://huggingface.co/RussRobin/SpatialBot-3B-LoRA) in HF, which is based on Phi-2 and SigLIP. You will need to modify paths in `config.json` to run on your device.
Merged and ready to run model is available at [SpatialBot-3B](https://huggingface.co/RussRobin/SpatialBot-3B).
Pretrained models can be found in [Model Zoo](https://github.com/BAAI-DCAI/Bunny?tab=readme-ov-file#model-zoo).

### Continuous  Fine-tuning

<details>
<summary>expand to see the instructions for continuously finetuing SpatialBot on your own data.</summary>


1. Prepare data: convert your data to a `JSON` file of a list of all samples with the format like:
```
[
    {
        'id': 'continuous_1',
        'image': ['/path/to/image_1','path_to_image_2'], # images are optional. We support 0-8 images, 0-2 recommended.
        "conversations": [
            {
                "from": "human",
                "value": "<image 1>\n<image 2>\nHello SpatialBot."
            },
            {
                "from": "gpt",
                "value": "Hi."
            }
    },
    {...}, # other QAs
]
```

2. Prepare model:

   * download merged LoRA [SpatialBot](https://huggingface.co/RussRobin/SpatialBot-3B) 

   * add `"continuous_training": true` in `/path/to/merged_model/config.json` to ensure loading the vision tower from merged weights
   
3. Edit script: both `finetune_full.sh` and `finetune_lora.sh` can be used, before:

   * change `--model_name_or_path` to `/path/to/merged_model`

   * delete `--pretrain_mm_mlp_adapter` because we load the cross-modality projector from merged weights

   * customize the hyperparameters, e.g. the learning rate, to fit your dataset

**Please note that** if you continuously fine-tune SpatialBot using LoRA, `--model-base` should be SpatialBot models rather than the original LLMs when loading.

</details>



## 🏆 SpatialBench
Please download [SpatialBench](https://huggingface.co/datasets/RussRobin/SpatialBench) and
put them under ```./eval/spatial_bench```.
Use our [SpatialBench script](https://github.com/BAAI-DCAI/SpatialBot/blob/main/script/eval/lora/spatial_bench.sh) to evaluate on it.

Parameters:
```--depth```: use this parameter is evaluating model with RGB-Depth input. Otherwise, RGB only. 

## 📃 SpatialBot Evaluation
Please follow our [general instructions LoRA](https://github.com/BAAI-DCAI/SpatialBot/blob/main/script/eval/lora/evaluation_lora.md),
or [general instructions Full-parameter](https://github.com/BAAI-DCAI/SpatialBot/blob/main/script/eval/full/evaluation_full.md)
to prepare data and evaluate SpatialBot on SpatialBench and general VLM benchmarks.

Please refer to [embodiment instructions](https://github.com/BAAI-DCAI/SpatialBot/blob/main/script/eval/lora/evaluation_embodiment.md)
to evaluate model on embodiment tasks.

To merge LoRA tuning models, see [merge instructions](https://github.com/BAAI-DCAI/SpatialBot/blob/main/script/merge_instruction.md)
## 🤔 CLI Inference
RGBD inference:
```
python -m bunny.serve.cli_depth \
	--model-path /path/to/bunny_lora_weights \
	--model-base /path/to/base_llm_model \
	--model-type phi-2 \ # NOTE: or phi-3/llama3-8b/qwen1.5-1.8b
	--conv-mode bunny \ # NOTE: or phi3/llama. bunny is for Phi-2 and QWen1.5
	--image-file /path/to/the/test/rgb/image \
	--depth-file /path/to/the/test/depth/image
```

RGB inference:
```
python -m bunny.serve.cli \
	--model-path /path/to/bunny_lora_weights \
	--model-base /path/to/base_llm_model \
	--model-type phi-2 \ # NOTE: or phi-3/llama3-8b/qwen1.5-1.8b
	--conv-mode bunny \ # NOTE: or phi3/llama. bunny is for Phi-2 and QWen1.5
	--image-file /path/to/the/test/rgb/image \
```

## 📊 SpatialQA Dataset

Please reach out to us if you are interested in SpatialQA: `wxcai@stanford.edu`.

Feel free to try [SpatialBot-3B model](https://huggingface.co/RussRobin/SpatialBot-3B), which is trained on SpatialQA.

[//]: # (### Image)

[//]: # (We use [LAION-2M]&#40;https://huggingface.co/datasets/BoyaWu10/Bunny-v1_0-data/tree/main/pretrain&#41; for pretraining. )

[//]: # (The finetuning dataset is based on [Bunny_695k]&#40;https://huggingface.co/datasets/BoyaWu10/Bunny-v1_0-data/tree/main/finetune&#41;. )

[//]: # (Please download images in Bunny_695k first, and then download [SpatialQA]&#40;https://huggingface.co/datasets/RussRobin/SpatialQA&#41;.)

[//]: # ()
[//]: # (### Data json)

[//]: # (Pretrain data json file can be found in [LAION-2M]&#40;https://huggingface.co/datasets/BoyaWu10/Bunny-v1_0-data/tree/main/pretrain&#41;.)

[//]: # ([SpatialQA]&#40;https://huggingface.co/datasets/RussRobin/SpatialQA&#41; is used in finetuning.)

### Prepare your own RGBD data
We recommend using depth information from sensors if possible.
Follow [depthmap instructions](https://github.com/BAAI-DCAI/SpatialBot/blob/main/SpatialQA_depthmap_instruction/SpatialQA_depthmap_instruction.md) to prepare estimated depth information on your own RGB images.

## ⚙ Embodied SpatialBot & SpatialQA-E Dataset
We collect SpatialQA-E, a robot manipulation dataset focusing on spatial understanding and reasoning. 

SpatialBot is finetuned on SpatialQA-E for pick-and-place abilities. It is a Vision-Language-Action (VLA) model, supporting multi-frame RGB or RGB-D inputs. 
One version of SpatialBot works by predicting delta position (or velocity) per each frame. 
Another version works by moving among predicted key points.
CKPTs and the dataset will be available soon.

A [preview](https://drive.google.com/drive/folders/1WBt5M0h2Z8k_ohPVUVEwIcCDaLxF9yv9?usp=sharing) of SpatialBot in embodiment is available.


## 🔗 Usage
If you find this repository helpful, please cite our paper.

```bibtex
@article{cai2024spatialbot,
  title={SpatialBot: Precise Spatial Understanding with Vision Language Models},
  author={Cai, Wenxiao and Ponomarenko, Yaroslav and Yuan, Jianhao and Li, Xiaoqi and Yang, Wankou and Dong, Hao and Zhao, Bo},
  journal={arXiv preprint arXiv:2406.13642},
  year={2024}
}
```

## 🧾 License
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg)](./LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC--BY--4.0-orange.svg)](./LICENSE)
[![Weight License](https://img.shields.io/badge/Weight%20License-CC--BY--4.0-red.svg)](./LICENSE)

The project employs specific datasets and checkpoints that are governed by their original licenses. Users must adhere to all terms and conditions outlined in these licenses. The checkpoints are restricted to uses that comply with the license agreements of Bunny, LLaMA 3, Phi-2, Phi-3, QWen-1.5, and GPT-4. The dataset is provided under the CC-BY-4.0 license.


## 📫 Acknowledgement

- The training of this work is built upon the [Bunny: A family of lightweight multimodal models](https://github.com/BAAI-DCAI/Bunny).
- This work utilizes LLMs from [Phi-2](https://huggingface.co/microsoft/phi-2), [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct), [QWen-1.5-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B) ,[QWen-1.5-4B](https://huggingface.co/Qwen/Qwen1.5-4B) , and [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
