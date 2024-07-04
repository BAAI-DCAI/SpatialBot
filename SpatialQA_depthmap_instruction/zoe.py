import torch
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
import glob
from argparse import ArgumentParser

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import save_raw_16bit,colorize

def zoe_run(zoe_model,img_path,save_path):
    image = Image.open(img_path).convert("RGB")
    depth = zoe_model.infer_pil(image)
    return save_raw_16bit(depth, save_path)

def zoe_from_json(json_file,img_dir,save_dir):
    with open(json_file, 'r') as file:
            data = json.load(file)
    os.makedirs(save_dir,exist_ok=True)

    for dt in tqdm(data, desc="Processing"):
        if type(dt['image']) is list:
            img_name = dt['image'][0].split('/')[-1] # maybe png, jpg, or ...
        else:
            img_name = dt['image'].split('/')[-1] # maybe png, jpg, or ...
        img_path = os.path.join(img_dir,img_name)
        img_name = img_name.split('.')[0]+'.png' # force png
        save_path = os.path.join(save_dir,img_name)
        zoe(zoe_model,img_path,save_path)

def zoe_dir(img_dir,save_dir,zoe_model,chunk=0,total_chunk=0,total_img=0):
    os.makedirs(save_dir,exist_ok=True)
    over_unit16 = 0

    image_extensions = ["jpg", "jpeg", "png", "gif", "bmp"]
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(img_dir, f"*.{extension}")))

    image_files = sorted(image_files)

    if (total_img != 0) and (total_chunk!=0):
        if total_img==-1:
            total_img = len(image_files)
        split_img = int(total_img/total_chunk) + 1
        start = split_img*chunk
        end = min(split_img*(chunk+1),len(image_files))
    else:
        start = 0
        end = len(image_files)

    for img_name in tqdm(image_files[start:end], desc="Processing"):
        img_name = img_name.split('/')[-1]
        img_path = os.path.join(img_dir,img_name)
        img_save_name = img_name.split('.')[0]+'.png' # force png

        save_path = os.path.join(save_dir,img_save_name)
        if os.path.exists(save_path):
            continue
        zoe_run(zoe_model,img_path,save_path)

def main(img_dir,save_dir,chunk=0,total_chunk=0,total_img=0):
    conf = get_config("zoedepth_nk", "infer")
    model_zoe_nk = build_model(conf)

    if(total_img != 0) and (total_chunk!=0):
        DEVICE = "cuda:"+str(chunk) 
    else:
        DEVICE = "cuda"
    zoe_model = model_zoe_nk.to(DEVICE)

    zoe_dir(img_dir,save_dir,zoe_model,chunk=chunk,total_chunk=total_chunk,total_img=total_img)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img-dir', type=str, default='/mnt/hpfs/baaidcai/bunny/data/finetune/images/kitti')
    parser.add_argument('--save-dir', type=str, default='/mnt/hpfs/baaidcai/bunny/data/finetune/images/kitti_d')
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--total-chunk", type=int, default=0)
    parser.add_argument('--total-img', type=int, default=0)

    args = parser.parse_args()
    main(img_dir = args.img_dir, save_dir = args.save_dir, chunk= args.chunk,total_chunk=args.total_chunk,total_img=args.total_img)