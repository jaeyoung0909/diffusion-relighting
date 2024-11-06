import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import glob
import os
from PIL import Image
from tqdm import tqdm


VALID_CAMERA_POSE_TYPE = ["NA3", "NE7", "CB5", "CF8", "NA7", "CC7", "CA2", "NE1", "NC3", "CE2"]
VALID_CAMERA_POSE_TYPE = {j : i for i, j in enumerate(VALID_CAMERA_POSE_TYPE)}
        
class LightDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = os.path.abspath(data_path)
        self.data = []
        objs_path = glob.glob(f'{self.data_path}/*')
        for obj_path in tqdm(objs_path, total=len(objs_path)):
            folder_name = os.path.basename(obj_path)
            obj_idx = folder_name.split('_')[1]
            obj_caption = folder_name.split('_')[2:]
            obj_caption = ' '.join(obj_caption)
            for camera_pose_type in VALID_CAMERA_POSE_TYPE.keys():
                for light_type in range(1, 14):
                    image_base_path = os.path.join(f"{obj_path}", "Lights", f"{light_type:03}", "transformed")
                    image_path = os.path.join(image_base_path, f'{camera_pose_type}.JPG')
                    image = Image.open(image_path)
                    self.data.append({
                        "obj_idx": int(obj_idx),
                        "obj_caption": obj_caption,
                        "pixel_values": image,
                        "light_type": int(light_type),
                        "camera_pose_type": VALID_CAMERA_POSE_TYPE[camera_pose_type],
                    })
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform:
            item = self.transform(item)
        return item


def make_train_dataset(args, tokenizer, accelerator):
    caption_column = "obj_caption"
    image_column = "pixel_values"


    def tokenize_captions(examples, is_train=True):
        # captions = []
        # for caption in examples[caption_column]:
        #     if random.random() < args.proportion_empty_prompts:
        #         captions.append("")
        #     elif isinstance(caption, str):
        #         captions.append(caption)
        #     elif isinstance(caption, (list, np.ndarray)):
        #         # take a random caption if there are multiple
        #         captions.append(random.choice(caption) if is_train else caption[0])
        #     else:
        #         raise ValueError(
        #             f"Caption column `{caption_column}` should contain either strings or lists of strings."
        #         )
        captions = examples[caption_column]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        # images = [image.convert("RGB") for image in examples[image_column]]
        # images = [image_transforms(image) for image in images]
        image = examples[image_column].convert("RGB")
        image = image_transforms(image)

        examples["pixel_values"] = image
        examples["input_ids"] = tokenize_captions(examples)
        examples['light_type'] = torch.full((1,), examples['light_type'], dtype=torch.long)
        examples['camera_pose_type'] = torch.full((1,), examples['camera_pose_type'], dtype=torch.long)

        return examples
    
    dataset = LightDataset(args.dataset_name, transform=preprocess_train)
    return dataset


def image_grid(imgs, rows, cols):
    """
    Concatenates multiple images
    """
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


if __name__=="__main__":
    dataset = LightDataset('data/lighting_patterns')
    import ipdb; ipdb.set_trace()