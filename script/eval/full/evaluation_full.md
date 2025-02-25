# Evaluation for full-parameter tuning models

We recommend trying RGBD on MME and SpatialBench, as they involve some depth related questions.

**Please change `conv-mode` to `minicpm/phi3/llama` for `MODEL_TYPE = phi-3/llama3-8b/qwen1.5-0.5b/qwen1.5-1.8b`.**

## SpatialBench
1. Download [SpatialBench](https://huggingface.co/datasets/RussRobin/SpatialBench) and put them under ```./eval/spatial_bench```.
2. The [script](https://github.com/BAAI-DCAI/SpatialBot/blob/main/script/eval/full/spatial_bench.sh) uses RGBD by default. To test with RGB, comment out ```--depth``` in it.
```shell
sh script/eval/full/spatial_bench.sh
```

## MME & MME-Depth

1. Refer to [MME GitHub](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) to download the benchmark dataset and put `MME_Benchmark_release_version` under `eval/mme`.
2. Update `MODEL_TYPE` and `TARGET_DIR` accordingly.
3. Run following lines if you want to run with RGB or RGBD
```shell
CUDA_VISIBLE_DEVICES=0 sh script/eval/full/mme.sh # RGB
CUDA_VISIBLE_DEVICES=0 sh script/eval/full/mme_depth.sh # RGBD
```

The responses and scores can be found in `eval/mme/answers_upload`.

## MMBench

1. Refer to [MMBench GitHub](https://github.com/open-compass/MMBench) to download the benchmark dataset. We support `MMBench-Dev`, `MMBench-Test`, `MMBench-Dev (cn)` and `MMBench-Test (cn)`. Please note that only the files downloaded by **legacy link** are supported.
   Put `MMBench_DEV_EN_legacy.tsv`, `MMBench_TEST_EN_legacy.tsv`, `MMBench_DEV_CN_legacy.tsv` or `MMBench_TEST_CN_legacy.tsv` under `eval/mmbench`.
2. Update `SPLIT`, `LANG (en/cn)`, `MODEL_TYPE` and `TARGET_DIR` accordingly.

```shell
CUDA_VISIBLE_DEVICES=0 sh script/eval/full/mmbench.sh
```

The response file can be found in `eval/mmbench/answers_upload`. You can submit the Excel file to [submission link](https://mmbench.opencompass.org.cn/mmbench-submission) to obtain the evaluation scores.

## SEED-Bench-1

1. Refer to [SEED-Bench Instruction](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md#data-preparation-for-seed-bench-1) to download the images and videos and put the images under `eval/seed-bench/SEED-Bench-image` and the videos under `eval/seed-bench/SEED-Bench-video`. Then, extract the video frames in the middle from the downloaded videos by running:

   ```shell
   pip install av decord
   python eval/seed-bench/extract_video_frames.py
   ```


2. Update `MODEL_TYPE` and `TARGET_DIR` accordingly.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash script/eval/full/seedbench.sh
```

The response file can be found in `eval/seed-bench/answers_upload` and the scores can be found in `eval/seed-bench/scores`.

## VQAv2

1. Download [COCO 2015 Test images](http://images.cocodataset.org/zips/test2015.zip) and put `test2015` under `eval/vqav2`. Then:

   ```shell
   tar -zxvf eval/vqav2/bunny_vqav2_mscoco_test2015.tar.gz -C eval/vqav2 && rm eval/vqav2/bunny_vqav2_mscoco_test2015.tar.gz && tar -zxvf eval/vqav2/bunny_vqav2_mscoco_test-dev2015.tar.gz -C eval/vqav2 && rm eval/vqav2/bunny_vqav2_mscoco_test-dev2015.tar.gz
   ```

2. Update `MODEL_TYPE` and `TARGET_DIR` accordingly.

```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash script/eval/full/vqav2.sh
```

The response file can be found in `eval/vqav2/answers_upload`. You can submit the `json` response file to [submission link](https://eval.ai/web/challenges/challenge-page/830) (Test-Dev Phase) to obtain the evaluation scores.

## GQA & GQA-Depth

1. Download the [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) of GQA, unzip it and put `images` under `eval/gqa`. Then:

   ```shell
   tar -zxvf eval/gqa/testdev_balanced_questions.tar.gz -C eval/gqa && rm eval/gqa/testdev_balanced_questions.tar.gz
   ```

2. Update `MODEL_TYPE` and `TARGET_DIR` accordingly.

```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash script/eval/full/gqa.sh # RGB
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash script/eval/full/gqa_depth.sh # RGBD
```

## POPE

1. Download [COCO 2014 Val images](http://images.cocodataset.org/zips/val2014.zip) and put `val2014` under `eval/pope`. Then, refer to [POPE GitHub](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) to download the benchmark dataset and put the three `json` files under `eval/pope/coco`.
2. Update `MODEL_TYPE` and `TARGET_DIR` accordingly.

```Shell
CUDA_VISIBLE_DEVICES=0 sh script/eval/full/pope.sh
```

We report the averaged F1-score of three categories (random, popular and adversarial).