import re
import random
import numpy as np
import os
import json
import yaml
import torch

from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from argparse import ArgumentParser

from bunny.model.builder import load_pretrained_model
from bunny.util.mm_utils import get_model_name_from_path, tokenizer_image_token, tokenizer_multi_image_token
from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates

from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt

import time

from collections import deque
import copy

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def call_bunny_engine_df(args, sample, model, tokenizer=None, processor=None):
    prompt = f"USER:{sample['question']} ASSISTANT:"
    input_ids = tokenizer_multi_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()

    for i in range(len(sample['image'])):
        sample['image'][i] = sample['image'][i].half()
    
    if sample['image'] is not None:
        output_ids = model.generate(
            input_ids,
            images = sample['image'],
            # images=[image1, image2],
            # images=[image1],
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=128,
            use_cache=True)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

    else:
        response = 'no image'
    return response


def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_answer(response,dataset):
    response_ = response.strip()[1:-1].replace(" ", "")
    response_ = response_.split(",")
    response_ = [float(res) for res in response_]   

    if dataset == 'maniskill':
        max_world = np.float32(0.2)
        min_world = np.float32(-0.2)
        max_angle = np.float32(0.2)
        min_angle = np.float32(-0.2)        
        for i in range(1, 4):
            offset = float(response_[i])
            response_[i] = offset * (max_world - min_world) + min_world        
        for i in range(4, 7):
            offset = float(response_[i])
            response_[i] = offset * (max_angle - min_angle) + min_angle
    else:
        raise NotImplementedError

    print('processed answer: ', response_)
    return response_

def main():
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument('--data-path', type=str, default=None)

    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--conv-mode", type=str, default='bunny')
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--answer-path", type=str, default=None)
    parser.add_argument("--goal-path", type=str, default=None)

    parser.add_argument("--dataset", type=str, default='maniskill')
    parser.add_argument("--buffer-len", type=int, default=4)

    args = parser.parse_args()
    assert args.image_path is not None
    assert args.answer_path is not None

    set_seed(42)

    processor = None
    call_model_engine = call_bunny_engine_df
    
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, vis_processors, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                                        args.model_type)

    img_buffer = deque(maxlen=args.buffer_len)

    print('bunny waiting for image from ros...')
    while True:
        if os.path.exists(args.image_path):
            print('bunny gets ros image')
            cur_time = time.time()
            try:
                image = Image.open(args.image_path).convert('RGB')
            except:
                time.sleep(0.1)
                continue
            image = vis_processors.preprocess(image, return_tensors='pt')['pixel_values'][0].to(device)
            img_buffer.append(image)
            
            img_list_converted = list(img_buffer)[::-1]
            if len(img_list_converted) < args.buffer_len:
                append_num = args.buffer_len-len(img_list_converted)
                for append_idx in range(append_num):
                    img_list_converted.append(copy.deepcopy(img_list_converted[-1]))

            while True:
                if os.path.exists(args.goal_path):
                    try:
                        with open(args.goal_path, "r", encoding="utf-8") as f:
                            goal = json.load(f)
                            break
                    except:
                        time.sleep(0.05)
                else:
                    time.sleep(0.1)
            question =  '<image 1>\n<image 2>\n<image 3>\n<image 4>\nInstruct the robot to '+ str(goal[0])+'\nAnswer with robot parameters.'
            print('question: ',question.split('\n')[-2])
            sample = {'question':question, 'image':img_list_converted} 

            with torch.no_grad():
                answer = call_model_engine(args, sample, model, tokenizer, processor)
                print('answer',answer)
                with open(args.answer_path, "w", encoding="utf-8") as f:
                    json.dump(answer, f)
                try:
                    os.remove(args.image_path)
                except OSError as e:
                    print("ros图片删除失败:", e)
                    raise OSError
            print('bunny finishes, in ', time.time()-cur_time,', and waiting for another ros image...')
            time.sleep(0.1)
        else:
            time.sleep(0.5)


if __name__ == '__main__':
    main()