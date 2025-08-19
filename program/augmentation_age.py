import os
from PIL import Image
from torchvision import transforms
import random

# --- CONFIG ---
AGE_DIR = 'datasets/clean/age'
IMG_SIZE = (128, 128)
AUGMENT_PER_IMAGE = 1 # number of augmented versions per original image

# Define augmentation transforms
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
])

# Iterate over age classes
for age_class in os.listdir(AGE_DIR):
    class_dir = os.path.join(AGE_DIR, age_class)
    if not os.path.isdir(class_dir):
        continue

    for fname in os.listdir(class_dir):
        if not fname.endswith('.jpg'):
            continue

        img_path = os.path.join(class_dir, fname)
        img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)

        # Create augmented images
        for i in range(AUGMENT_PER_IMAGE):
            aug_img = augment_transform(img)
            aug_fname = fname.replace('.jpg', f'_aug{i}.jpg')
            aug_img.save(os.path.join(class_dir, aug_fname))

print("Age data augmentation completed successfully!")