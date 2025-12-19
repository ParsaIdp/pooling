#!/usr/bin/env python3
"""
Script to download and prepare Tiny ImageNet-200.
"""

import os
import zipfile
import subprocess
from torchvision.datasets.utils import download_url

def prepare_tiny_imagenet(root="./data", clean=False):
    dataset_dir = os.path.join(root, "tiny-imagenet-200")
    if os.path.exists(dataset_dir) and not clean:
        print(f"Dataset already exists at {dataset_dir}")
        return

    # 1. Download
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    print("Downloading Tiny ImageNet...")
    download_url(url, root, "tiny-imagenet-200.zip", None)

    # 2. Extract
    print("Extracting...")
    with zipfile.ZipFile(os.path.join(root, "tiny-imagenet-200.zip"), 'r') as zip_ref:
        zip_ref.extractall(root)

    # 3. Format Validation Set
    # Tiny ImageNet val set is just a list of images in `val/images/`
    # and a `val/val_annotations.txt` file mapping image -> class.
    # PyTorch ImageFolder requires `val/class_name/image.jpg`.
    print("Formatting validation set...")
    val_dir = os.path.join(dataset_dir, "val")
    val_img_dir = os.path.join(val_dir, "images")
    
    # Read annotations
    with open(os.path.join(val_dir, "val_annotations.txt"), 'r') as f:
        lines = f.readlines()
    
    val_img_to_class = {}
    for line in lines:
        parts = line.strip().split('\t')
        val_img_to_class[parts[0]] = parts[1]
    
    # Move images to class subfolders
    for img_file, class_name in val_img_to_class.items():
        src = os.path.join(val_img_dir, img_file)
        dst_dir = os.path.join(val_dir, class_name)
        dst = os.path.join(dst_dir, img_file)
        
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.exists(src):
            os.rename(src, dst)
    
    # Clean up empty images folder
    os.rmdir(val_img_dir)
    print("Done!")

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    prepare_tiny_imagenet()
