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


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def call_bunny_engine_df(args, sample, model, tokenizer=None, processor=None):
    conv = conv_templates[args.conv_mode].copy()
    prompt = conv.system+f"USER:{sample['question']} ASSISTANT:"

    input_ids = tokenizer_multi_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()

    if type(sample['image']) is not list:
        sample['image'] = [sample['image']]
    for i in range(len(sample['image'])):
        sample['image'][i] = sample['image'][i].half()
    
    if sample['image'] is not None:
        output_ids = model.generate(
            input_ids,
            images = sample['image'],
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

def main():
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument('--config-path', type=str, default=None)
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--output-path', type=str, default=None)

    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--conv-mode", type=str, default='bunny')
    parser.add_argument("--question", type=str, default=None)

    args = parser.parse_args()

    set_seed(42)

    processor = None
    call_model_engine = call_bunny_engine_df
    
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, vis_processors, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                                          args.model_type)
    data_json = json.load(open(os.path.join(args.data_path,args.question.strip())))

    dataset = []
    for idx in range(len(data_json)):
        # if idx<10000: continue
        if idx>100: break
        data = data_json[idx]
        for i in range(len(data['image'])):
            data['image'][i] = Image.open(os.path.join(args.data_path,'images',data['image'][i]))
        dataset.append({'id': data['id'], 'question': data['conversations'][0]['value'], 'answer': data['conversations'][1]['value'],
                'image': data['image']})

    x_values = []
    y_values = []
    z_values = []
    x_gt = []
    y_gt = []
    z_gt = []

    angle1_values = []
    angle2_values = []
    angle3_values = []
    angle1_gt = []
    angle2_gt = []
    angle3_gt = []

    for sample in dataset:
        for i in range(len(sample['image'])):
            sample['image'][i] = vis_processors.preprocess(sample['image'][i], return_tensors='pt')['pixel_values'][0].to(device)

        with torch.no_grad():
            response = call_model_engine(args, sample, model, tokenizer, processor)
            print('question:',sample['question'],'response',response,'gt',sample['answer'])

            xyz_positions = [1,2,3,4,5,6]
            try:
                response_ = response.strip()[1:-1].replace(" ", "")
                response_ = response_.split(",")
                response_ = [int(res) for res in response_]   
                x_values.append(response_[xyz_positions[0]])
                y_values.append(response_[xyz_positions[1]])
                z_values.append(response_[xyz_positions[2]])
                angle1_values.append(response_[xyz_positions[3]])
                angle2_values.append(response_[xyz_positions[4]])
                angle3_values.append(response_[xyz_positions[5]])
                
                answer = sample['answer']
                answer = answer.strip()[1:-1].replace(" ", "")
                answer = answer.split(",")
                answer = [int(res) for res in answer]   
                x_gt.append(answer[xyz_positions[0]])
                y_gt.append(answer[xyz_positions[1]])
                z_gt.append(answer[xyz_positions[2]])

                angle1_gt.append(answer[xyz_positions[3]])
                angle2_gt.append(answer[xyz_positions[4]])
                angle3_gt.append(answer[xyz_positions[5]])
            except:
                try:
                    response_ = response.strip()[1:-1].replace(" ", "")
                    response_ = response_.split(",")
                    response_ = [float(res) for res in response_]           
                    x_values.append(response_[xyz_positions[0]])
                    y_values.append(response_[xyz_positions[1]])
                    z_values.append(response_[xyz_positions[2]])

                    angle1_values.append(response_[xyz_positions[3]])
                    angle2_values.append(response_[xyz_positions[4]])
                    angle3_values.append(response_[xyz_positions[5]])

                    answer = sample['answer']
                    answer = answer.strip()[1:-1].replace(" ", "")
                    answer = answer.split(",")
                    answer = [float(res) for res in answer]   
                    x_gt.append(answer[xyz_positions[0]])
                    y_gt.append(answer[xyz_positions[1]])
                    z_gt.append(answer[xyz_positions[2]])

                    angle1_gt.append(answer[xyz_positions[3]])
                    angle2_gt.append(answer[xyz_positions[4]])
                    angle3_gt.append(answer[xyz_positions[5]])
                except:
                    continue
    
    plt.figure(figsize=(20, 6))
    plt.subplot(3, 2, 1) 
    plt.plot(x_values, label='x')
    plt.plot(x_gt, label='x_gt')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(y_values, label='y')
    plt.plot(y_gt, label='y_gt')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(z_values, label='z')
    plt.plot(z_gt, label='z_gt')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(angle1_values, label='angle 1')
    plt.plot(angle1_gt, label='angle1_gt')
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(angle2_values, label='angle 2')
    plt.plot(angle2_gt, label='angle2_gt')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(angle3_values, label='angle 3')
    plt.plot(angle3_gt, label='angle3_gt')
    plt.legend()

    plt.tight_layout() 
    plt.savefig('control.png')

if __name__ == '__main__':
    main()