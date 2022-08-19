import argparse
import os

import cv2
import numpy as np
from PIL import Image

from utils.common import tti
from utils.demo import load_data, create_pipeline
import shutil

current_folder = os.path.dirname(__file__)

def inpaint(output_folder, folder_name, checkpoint_path, tmp_folder):
    for view in ["front", "left", "right", "back"]:
        source_sample = folder_name
        target_sample = folder_name +"_%s" % view
        device = "cuda:0"
    
        data_dict = load_data(tmp_folder, source_sample, target_sample, device=device)
    
        pipeline = create_pipeline(checkpoint_path, device=device)
        output_dict = pipeline(data_dict)
    
        pred_img = (tti(output_dict['refined']) * 255).astype(np.uint8)
    
        pair_str = source_sample + '_to_' + target_sample
        target_img_path = os.path.join(tmp_folder, "target_img", pair_str + '.png')
        
        print('Writing the .png result to:', target_img_path)
        cv2.imwrite(target_img_path, pred_img[..., ::-1])
        
        
        output_image_path = os.path.join(output_folder, folder_name, "body_parts_%s_texture.png" % view)
        output_uv_path = os.path.join(output_folder, folder_name, "uv_%s.npy" % view)
        
        shutil.copy(target_img_path, output_image_path)
        
        image = Image.open(output_image_path)
        uv = np.load(output_uv_path)
        mask = uv[:, :, 0]
        mask[~np.isnan(mask)] = 1
        mask[np.isnan(mask)] = 0
        
        mask = np.flip(mask, 1)
        mask = np.swapaxes(mask, 0, 1)  
        
        mask = np.expand_dims(mask.astype(np.uint8), axis=2)
        white = np.ones_like(mask) * 255
        stacked_mask = np.concatenate([mask, mask, mask], axis=2)
        white_background = np.concatenate([white, white, white], axis=2)
        
        image_np = np.asarray(image, np.uint8)
        image_np = image_np * stacked_mask + white_background * stacked_mask
        image_np = np.concatenate([image_np, mask * 255], axis=2)
        
        masked_image = Image.fromarray(image_np)
        masked_image.save(output_image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", default=os.path.join(current_folder, 'data'))
    parser.add_argument("--tmp_folder", default=r"E:\CNN\implicit_functions\characters\tmp")
    parser.add_argument("--out_folder", default=r"E:\CNN\implicit_functions\characters\out")
    parser.add_argument("--sub_folders", default="")
    args = parser.parse_args()
    
    root_folder = args.out_folder   
    tmp_folder = os.path.join(args.tmp_folder, "textures")
    model_folder = os.path.join(args.model_folder, "checkpoint")    
    subfolder_names = os.listdir(root_folder) if args.sub_folders == "" else args.sub_folders.split("|")
    
    for folder_name in subfolder_names:
        print(folder_name)
        inpaint(root_folder, folder_name, model_folder, tmp_folder)