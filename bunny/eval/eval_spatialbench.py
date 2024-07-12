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
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--conv-mode", type=str, default='bunny')
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--depth", action='store_true')

    args = parser.parse_args()

    set_seed(42)

    processor = None
    call_model_engine = call_bunny_engine_df

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, vis_processors, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                                        args.model_type)
    data_json = json.load(open(os.path.join(args.data_path,args.question.strip())))

    dataset = []
    for idx in range(len(data_json)):
        data = data_json[idx]
        if type(data['image']) is str: 
            data['image'] = [data['image']]
            
        for i in range(len(data['image'])):
            if args.depth:
                dataset_name = data['image'][i].split('/')[0]+'_d'
                depth_path = os.path.join(args.data_path,dataset_name,data['image'][i].split('/')[-1].split('.')[0]+'.png')
            data['image'][i] = Image.open(os.path.join(args.data_path,data['image'][i]))
            if args.depth:
                data['image'].append(Image.open(depth_path))
        
        if 'id' not in data.keys():
            data['id'] = idx #  CWX NOTE counting, 等的json目前还没有id

        if str(data['id']).split('_')[0] == 'grounding':
            idx_2 = 0
            for obj_bbox in data['objects']:
                for obj_ in obj_bbox.keys():
                    question = 'Please provide the bounding box coordinate of the object this sentence describes: '+ obj_
                    gt = obj_bbox[obj_]
                    dataset.append({'id': data['id']+'_'+str(idx_2), 'question': question, 'answer': gt,
                        'image': data['image']})
                    idx_2 = idx_2 + 1
        else:
            question = ''.join([f'<image {j}>\n' for j in range(1, len(data['image'])+1)])+data['question']
            dataset.append({'id': data['id'], 'question': question, 'answer': data['answer'],
                    'image': data['image']})
                    
    j = 0
    responses = []
    for sample in dataset:
        for i in range(len(sample['image'])):
            sample['image'][i] = vis_processors.preprocess(sample['image'][i], return_tensors='pt')['pixel_values'][0].to(device)

        with torch.no_grad():
            response = call_model_engine(args, sample, model, tokenizer, processor)
            responses.append({'response':response,'gt':sample['answer']})
            j = j+1

    if args.question.strip() in ['reach_pos_neg.json','size_pos_neg.json']:
        scores = 0
        full_scores = 0
        for j in range(len(responses)):
            if j%2==1: continue
            pos_response = responses[j]['response']
            pos_gt = responses[j]['gt']
            neg_response = responses[j+1]['response']
            neg_gt = responses[j+1]['gt']
            accuracy = 0
            accuracy_plus = 0
            if pos_gt in pos_response:
                accuracy = accuracy + 1
            if neg_gt in neg_response:
                accuracy = accuracy + 1
            if (pos_gt in pos_response) and (neg_gt in neg_response):
                accuracy_plus = accuracy_plus + 1
            print(str(j),'responses',pos_response,' ',neg_response,' gt: ',pos_gt,' ',neg_gt, ' accuracy accuracy_plus: ',str(accuracy),' ',str(accuracy_plus))
            scores = scores + accuracy + accuracy_plus
            full_scores = full_scores + 3
        print(args.question.strip(),', get', str(scores),' out of ',str(full_scores), ' ',str(100*scores/full_scores),'%')
    elif args.question.strip() in ['positional.json','size.json']:
        scores = 0
        full_scores = 0
        for j in range(len(responses)):
            if responses[j]['gt'] in responses[j]['response']:
                scores = scores + 1
            full_scores = full_scores + 1
        print('get ', str(scores),' out of ',str(full_scores), ' ',str(100*scores/full_scores),'%')
    elif args.question.strip() in ['existence.json']: 
        scores = 0
        full_scores = 0
        for j in range(len(responses)):
            if j%2==1: continue
            pos_response = responses[j]['response']
            pos_gt = responses[j]['gt']
            neg_response = responses[j+1]['response']
            neg_gt = responses[j+1]['gt']
            if (pos_gt in pos_response) and (neg_gt in neg_response):
                scores = scores+1
            full_scores = full_scores + 1
        print('get ', str(scores),' out of ',str(full_scores), ' ',str(100*scores/full_scores),'%')
    elif args.question.strip() in ['counting.json']: 
        errors = []
        for j in range(len(responses)):
            error = abs(int(responses[j]['response'])-int(responses[j]['gt'])) / int(responses[j]['gt']) * 100
            errors.append(error)
        scores = 100 - sum(errors)/len(errors)
        print('get ', str(scores),' out of 100.')


if __name__ == '__main__':
    main()