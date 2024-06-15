import os
import random
import torch
import numpy as np
import pickle

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

PIL_INTERPOLATION = {
    "linear": Image.Resampling.BILINEAR,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "nearest": Image.Resampling.NEAREST,
}

VALID_IMG_FORMATS = ("jpeg", "png", "jpg", "webp")

class DiscoveryDataset(Dataset):
    def __init__(self,
            data_root_dir,
            tokenizer,
            size=512,
            repeats=100,
            interpolation="bicubic",
            flip_p=0.5,
            dataset_type="train",
            placeholder_tokens="",
            center_crop=False,
            ):

        self.data_root = data_root_dir
        self.tokenizer = tokenizer

        self.size = size
        self.placeholder_tokens = placeholder_tokens
        self.placeholder_tokens_ids = self.tokenizer.convert_tokens_to_ids(self.placeholder_tokens)
        
        self.center_crop = center_crop
        self.flip_p = flip_p

        # Traverse through all images
        self.image_paths = []
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                file_extension = file.split(".")[-1]
                if file_extension in VALID_IMG_FORMATS:
                    self.image_paths.append(os.path.join(root, file))
        
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if dataset_type == "train":
            self._length = self.num_images * repeats

        self.interpolation = PIL_INTERPOLATION[interpolation]
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def apply_center_crop(self, image):
        img = np.array(image).astype(np.uint8)
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        return Image.fromarray(img)

    def __getitem__(self, i):
        data_entry = dict()
        image = Image.open(self.image_paths[i])
        image = image.convert("RGB") if image.mode != "RGB" else image

        # Apply center crop if required
        if self.center_crop:
            image = self.apply_center_crop(image)

        # Final step of preprocessing
        image_processed = image.resize((self.size, self.size), resample=self.interpolation)
        image_processed = self.flip_transform(image_processed)
        image_processed = np.array(image_processed).astype(np.uint8)
        image_processed = (image_processed / 127.5 - 1.0).astype(np.float32) # Pixel values between -1, 1
        image_processed = torch.from_numpy(image_processed).permute(2, 0, 1)
        
        # Register the members of data entry
        data_entry["img_pixel_values"] = image_processed
        data_entry["img_path"] = self.image_paths[i]
        
        return data_entry


