<h1 align = "center">
  SpatialBot
</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2406.13642">
        <img alt="Paper" src="http://img.shields.io/badge/Paper-arXiv%3A2406.13642-B31B1B.svg">
    </a>
</p>

This is the official repo for "SpatialBot: Precise Spatial Understanding with Vision Language Models".

SpatialQA dataset, SpatialBench benchmark, and SpatialBot models and checkpoints will be available upon the release of the final version of the paper.

[comment]: <> (## 🤖 SpatialBot Model)

[comment]: <> (### Install)

[comment]: <> (Please refer to [Bunny]&#40;https://github.com/BAAI-DCAI/Bunny?tab=readme-ov-file#install&#41; to install locally, or start from docker image: `docker pull russellrobin/bunny:latest`)

## Model Installation
SpatialBot is a multi-image version of [Bunny](https://github.com/BAAI-DCAI/Bunny). 
If you've installed Bunny, just replace the code with ours are reinstall ```bunny``` package.
You can start from a docker or configure local environments.

### Start from Docker
We provide a ready to run environment. Just update it with our codes:
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
Follow instructions [here](https://github.com/BAAI-DCAI/Bunny?tab=readme-ov-file#local-installation), but use codes from this repo.


## 🔗 Citation
If you find this repository helpful, please cite the paper below.

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
