import json
import numpy as np
from PIL import Image
import random
import os
import glob
import argparse
import multiprocessing as mp

def transform_and_save(img_path, mask_path):
    if 'obj_64' in mask_path:
        mask_path = mask_path.replace('com_masks', 'obj_masks')
    CROP_SIZE = 512
    image = Image.open(img_path)
    mask = Image.open(mask_path).convert("L")

    # Image resize
    width, height = image.size
    shorter_size = min(width, height)
    scale_factor = 512 / shorter_size
    width, height = int(width * scale_factor), int(height * scale_factor)
    resized_image = image.resize((width, height))
    resized_mask = mask.resize((width, height))

    # Center crop (512 x 512)
    width, height = resized_image.size
    left = (width - CROP_SIZE) / 2
    top = (height - CROP_SIZE) / 2
    right = (width + CROP_SIZE) / 2
    bottom = (height + CROP_SIZE) / 2

    cropped_image = resized_image.crop((left, top, right, bottom))
    cropped_mask = resized_mask.crop((left, top, right, bottom))
    masked_image = Image.composite(
        cropped_image, Image.new("RGB", cropped_image.size, "black"), cropped_mask
    )
    output_path = img_path.replace('raw_undistorted', 'transformed')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    masked_image.save(output_path)

data_path = "data/lighting_patterns"
data_path = os.path.abspath(data_path)
VALID_CAMERA_POSE_TYPE = ["NA3", "NE7", "CB5", "CF8", "NA7", "CC7", "CA2", "NE1", "NC3", "CE2"]
VALID_CAMERA_POSE_TYPE = {j : i for i, j in enumerate(VALID_CAMERA_POSE_TYPE)}

objs_path = glob.glob(f'{data_path}/*')
image_pathes = []
mask_pathes = []
for obj_path in objs_path:
    folder_name = os.path.basename(obj_path)
    obj_idx = folder_name.split('_')[1]
    obj_caption = folder_name.split('_')[2:]
    for camera_pose_type in VALID_CAMERA_POSE_TYPE.keys():
        for light_type in range(1, 14):
            light_base_path = os.path.join(f"{obj_path}", "Lights", f"{light_type:03}", "raw_undistorted")
            image_path = os.path.join(light_base_path, f'{camera_pose_type}.JPG')
            image_pathes.append(image_path)
            mask_base_path = os.path.join(f"{obj_path}", "output", 'com_masks')
            mask_path = os.path.join(mask_base_path, f'{camera_pose_type}.png')
            mask_pathes.append(mask_path)

with mp.Pool(64) as pool:
    pool.starmap(transform_and_save, zip(image_pathes, mask_pathes))