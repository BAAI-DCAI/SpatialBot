<h1 align = "center">
  SpatialBot
</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2406.13642">
        <img alt="Paper" src="http://img.shields.io/badge/Paper-arXiv%3A2406.13642-B31B1B.svg">
    </a>
    <a href="https://huggingface.co/datasets/RussRobin/SpatialQA">
        <img alt="Dataset" src="https://img.shields.io/badge/🤗%20Dataset-SpatialQA-yellow">
    </a>
    <a href="https://huggingface.co/RussRobin/SpatialBot-3B">
        <img alt="Model SpatialBot-3B" src="https://img.shields.io/badge/🤗%20Model-SpatialBot--3B-green">
    </a>
    <a href="https://huggingface.co/datasets/RussRobin/SpatialBench">
        <img alt="Benchmark" src="https://img.shields.io/badge/🤗%20Benchmark-SpatialBench-blue">
    </a>
</p>

This is the official repo for "SpatialBot: Precise Spatial Understanding with Vision Language Models".

<!-- ![comparison_8B](comparison_8B.png) -->

## 📊 SpatialQA Dataset

### Image
We use [LAION-2M](https://huggingface.co/datasets/BoyaWu10/Bunny-v1_0-data/tree/main/pretrain) for pretraining. 
The finetuning dataset is based on [Bunny_695k](https://huggingface.co/datasets/BoyaWu10/Bunny-v1_0-data/tree/main/finetune). 
Please download images in Bunny_695k first, and then download [SpatialQA](https://huggingface.co/datasets/RussRobin/SpatialQA).

### Data json
Pretrain data json file can be found in [LAION-2M](https://huggingface.co/datasets/BoyaWu10/Bunny-v1_0-data/tree/main/pretrain).
[SpatialQA](https://huggingface.co/datasets/RussRobin/SpatialQA) is used in finetuning.

### Prepare your own RGBD data
We recommend using depth information from sensors if possible.
Follow [depthmap instructions](https://github.com/BAAI-DCAI/SpatialBot/blob/main/SpatialQA_depthmap_instruction/SpatialQA_depthmap_instruction.md) to prepare estimated depth information on your own RGB images.

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

Please find finetuned ckpts of [SpatialBot-3B-LoRA](https://huggingface.co/RussRobin/SpatialBot-3B-LoRA) in HF, which is based on Phi-2 and SigLIP.
Merged model is available at [SpatialBot-3B](https://huggingface.co/RussRobin/SpatialBot-3B).
Pretrained models can be found in [Model Zoo](https://github.com/BAAI-DCAI/Bunny?tab=readme-ov-file#model-zoo).


## 🏆 SpatialBench
Please download [SpatialBench](https://huggingface.co/datasets/RussRobin/SpatialBench) and
put them under ```./eval/spatialqa_bench```.
Use our [SpatialBench script](https://github.com/BAAI-DCAI/SpatialBot/blob/main/script/eval/lora/spatial_bench.sh) to evaluate on it.

Parameters:
```--depth```: use this parameter is evaluating model with RGB-Depth input. Otherwise, RGB only. 

## 📃 SpatialBot Evaluation
Please follow our [general instructions](https://github.com/BAAI-DCAI/SpatialBot/blob/main/script/eval/lora/evaluation_lora.md) 
to prepare data and evaluate SpatialBot on SpatialBench and general VLM benchmarks.

Please refer to [embodiment instructions](https://github.com/BAAI-DCAI/SpatialBot/blob/main/script/eval/lora/evaluation_embodiment.md)
to evaluate model on embodiment tasks.

## CLI Inference
RGBD inference:
```ssh
python -m bunny.serve.cli_depth \
	--model-path /path/to/bunny_lora_weights \
	--model-base /path/to/base_llm_model \
	--model-type phi-2 \ # NOTE: or phi-3/llama3-8b/qwen1.5-1.8b
	--conv-mode bunny \ # NOTE: or phi3/llama. bunny is for Phi-2 and QWen1.5
	--image-file /path/to/the/test/rgb/image \
	--depth-file /path/to/the/test/depth/image
```

RGB inference:
```ssh
python -m bunny.serve.cli \
	--model-path /path/to/bunny_lora_weights \
	--model-base /path/to/base_llm_model \
	--model-type phi-2 \ # NOTE: or phi-3/llama3-8b/qwen1.5-1.8b
	--conv-mode bunny \ # NOTE: or phi3/llama. bunny is for Phi-2 and QWen1.5
	--image-file /path/to/the/test/rgb/image \
```

## 🔗 Usage
If you find this repository helpful, please cite our paper.

```bibtex
@inproceedings{Cai2024SpatialBotPS,
  title={SpatialBot: Precise Spatial Understanding with Vision Language Models},
  author={Wenxiao Cai and Yaroslav Ponomarenko and Jianhao Yuan and Xiaoqi Li and Wankou Yang and Hao Dong and Bo Zhao},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:270619467}
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
