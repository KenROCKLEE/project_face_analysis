"""
UTKFace Age and Gender Prediction Pipeline in PyTorch
Author: Your Name
Date: 2025-08-20
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image

# ----------------- CONFIG -----------------
IMG_SIZE = (128, 128)
AGE_DIR = 'datasets/raw/UTKFace'   # Directory for age data (raw)
GENDER_DIR = 'datasets/clean/gender'  # Directory for gender data (cleaned)
MODEL_DIR = 'models'
BATCH_SIZE = 32
EPOCHS_AGE = 50
EPOCHS_GENDER = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_AGE = 116  # maximum age in UTKFace dataset

# Gender classes
GENDER_CLASSES = sorted([d for d in os.listdir(GENDER_DIR) if os.path.isdir(os.path.join(GENDER_DIR, d))])
NUM_GENDER_CLASSES = len(GENDER_CLASSES)
GENDER_MAP = {folder: idx for idx, folder in enumerate(GENDER_CLASSES)}

# Data normalization (important for pre-trained models like ResNet)
normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Transforms for training and validation
standard_transforms = transforms.Compose([
    transforms.ToTensor(),
    normalize_transform
])

# ----------------- UTILS -----------------
def load_dataset(age_dir=AGE_DIR, gender_dir=GENDER_DIR, img_size=IMG_SIZE):
    X_age, y_age = [], []
    X_gender, y_gender = [], []

    print("Loading age data from raw directory...")
    for fname in os.listdir(age_dir):
        if fname.endswith('.jpg'):
            try:
                parts = fname.split('_')
                age = int(parts[0])
                img_path = os.path.join(age_dir, fname)
                img = Image.open(img_path).convert('RGB').resize(img_size)
                X_age.append(img)
                y_age.append(age / MAX_AGE)  # normalize age to [0,1]
            except (ValueError, IndexError):
                continue
    
    print("Loading gender data from cleaned directory...")
    for folder_name in GENDER_CLASSES:
        class_dir = os.path.join(gender_dir, folder_name)
        gender_label = GENDER_MAP[folder_name]
        for fname in os.listdir(class_dir):
            if fname.endswith('.jpg'):
                img_path = os.path.join(class_dir, fname)
                img = Image.open(img_path).convert('RGB').resize(img_size)
                X_gender.append(img)
                y_gender.append(gender_label)
    
    return np.array(X_age), np.array(y_age), np.array(X_gender), np.array(y_gender)

def split_dataset(X_age, y_age, X_gender, y_gender, test_size=0.2, val_size=0.1, random_state=42):
    X_age_train, X_age_test, y_age_train, y_age_test = train_test_split(X_age, y_age, test_size=test_size, random_state=random_state)
    X_age_train, X_age_val, y_age_train, y_age_val = train_test_split(X_age_train, y_age_train, test_size=val_size, random_state=random_state)

    X_gender_train, X_gender_test, y_gender_train, y_gender_test = train_test_split(X_gender, y_gender, test_size=test_size, random_state=random_state)
    X_gender_train, X_gender_val, y_gender_train, y_gender_val = train_test_split(X_gender_train, y_gender_train, test_size=val_size, random_state=random_state)

    return X_age_train, X_age_val, X_age_test, y_age_train, y_age_val, y_age_test, \
           X_gender_train, X_gender_val, X_gender_test, y_gender_train, y_gender_val, y_gender_test

class AgeDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]  # already normalized [0,1]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

class GenderDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

def create_dataloader(X, y, batch_size=BATCH_SIZE, task='age', transforms=None):
    if task == 'age':
        dataset = AgeDataset(X, y, transform=transforms)
    else:
        dataset = GenderDataset(X, y, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ----------------- MODEL -----------------
class ResNetModel(nn.Module):
    def __init__(self, output_units=1, task='age'):
        super().__init__()
        
        # Load pre-trained ResNet-18 model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze all layers first
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Unfreeze the last block (layer4) and the fully connected layer
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        
        # Replace the final classification layer with a new one for your task
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, output_units)
        self.task = task

    def forward(self, x):
        return self.resnet(x)

# ----------------- TRAIN -----------------
def train_model(model, train_loader, val_loader, task='age', epochs=50, lr=1e-3, model_name='model.pth'):
    model.to(DEVICE)
    if task == 'age':
        criterion = nn.L1Loss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses, val_losses, val_metrics = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb).squeeze()

            if task == 'age':
                loss = criterion(output, yb)
            else:
                loss = criterion(output, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train = running_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_metric = 0 
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                output = model(xb).squeeze()
                
                if task == 'age':
                    loss = criterion(output, yb)
                    val_metric += (torch.abs(output - yb).mean().item() * MAX_AGE)  # MAE in years
                else:
                    loss = criterion(output, yb)
                    pred = (torch.sigmoid(output) > 0.5).float()
                    val_metric += (pred == yb).sum().item()
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        if task == 'age':
            avg_val_metric = val_metric / len(val_loader)
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE (years): {avg_val_metric:.4f}")
        else:
            avg_val_metric = val_metric / len(val_loader.dataset)
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_metric:.4f}")

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            print(f"Early stopping at epoch {epoch} due to no improvement in Val Loss.")
            break

        train_losses.append(avg_train)
        val_losses.append(avg_val_loss)
        val_metrics.append(avg_val_metric)

    os.makedirs(MODEL_DIR, exist_ok=True)
    if best_model_state:
        torch.save(best_model_state, os.path.join(MODEL_DIR, model_name))
        print(f"Model saved from best epoch with Val Loss: {best_val_loss:.4f}")
    else:
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_name))

    # Save plots
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{task} Training Loss')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, f'{task}_loss_curve.png'))
    plt.close()

    plt.figure(figsize=(8,4))
    metric_label = 'Validation MAE (years)' if task=='age' else 'Validation Accuracy'
    plt.plot(val_metrics, label=metric_label)
    plt.title(f'{task} Validation Metric')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, f'{task}_metric_curve.png'))
    plt.close()

# ----------------- EVALUATION -----------------
def evaluate_model(model, test_loader, task='age'):
    model.to(DEVICE)
    model.eval()
    total_metric = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb).squeeze()
            if task == 'age':
                mae = (torch.abs(output - yb).mean().item() * MAX_AGE)
                total_metric += mae
            else:
                pred = (torch.sigmoid(output) > 0.5).float()
                acc = (pred == yb).sum().item()
                total_metric += acc
    if task == 'age':
        print(f"Test MAE (years): {total_metric / len(test_loader):.4f}")
    else:
        print(f"Test Accuracy: {total_metric / len(test_loader.dataset):.4f}")

# ----------------- MAIN -----------------
def main():
    if torch.cuda.is_available():
        print(f"✅ GPU is available and will be used! Device: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ GPU not available, using CPU. This will be slow.")

    print("Loading dataset...")
    X_age, y_age, X_gender, y_gender = load_dataset()
    print(f"Total age samples: {len(X_age)} | Total gender samples: {len(X_gender)}")

    X_age_train, X_age_val, X_age_test, y_age_train, y_age_val, y_age_test, \
    X_gender_train, X_gender_val, X_gender_test, y_gender_train, y_gender_val, y_gender_test = \
        split_dataset(X_age, y_age, X_gender, y_gender)

    # Dataloaders
    age_train_loader = create_dataloader(X_age_train, y_age_train, task='age', transforms=standard_transforms)
    age_val_loader = create_dataloader(X_age_val, y_age_val, task='age', transforms=standard_transforms)
    age_test_loader = create_dataloader(X_age_test, y_age_test, task='age', transforms=standard_transforms)
    
    gender_train_loader = create_dataloader(X_gender_train, y_gender_train, task='gender', transforms=standard_transforms)
    gender_val_loader = create_dataloader(X_gender_val, y_gender_val, task='gender', transforms=standard_transforms)
    gender_test_loader = create_dataloader(X_gender_test, y_gender_test, task='gender', transforms=standard_transforms)

    # Train Age model
    print("Training Age model with ResNet...")
    age_model = ResNetModel(output_units=1, task='age')
    train_model(age_model, age_train_loader, age_val_loader, task='age', epochs=EPOCHS_AGE, model_name='age_model_resnet.pth')
    evaluate_model(age_model, age_test_loader, task='age')

    # Train Gender model
    print("Training Gender model with ResNet...")
    gender_model = ResNetModel(output_units=1, task='gender')
    train_model(gender_model, gender_train_loader, gender_val_loader, task='gender', epochs=EPOCHS_GENDER, model_name='gender_model_resnet.pth')
    evaluate_model(gender_model, gender_test_loader, task='gender')

    print("All models trained, evaluated, and saved successfully!")

if __name__ == '__main__':
    main()
