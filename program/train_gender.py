"""
UTKFace Gender Prediction Pipeline in PyTorch
Author: Your Name
Date: 2025-08-23
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

# ----------------- CONFIG -----------------
IMG_SIZE = (128, 128)
GENDER_DIR = 'datasets/clean/gender'
MODEL_DIR = 'models'
BATCH_SIZE = 32
EPOCHS_GENDER = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EARLY_STOPPING_PATIENCE = 7  # stop if val loss doesn't improve

# Gender classes
GENDER_CLASSES = sorted([d for d in os.listdir(GENDER_DIR) if os.path.isdir(os.path.join(GENDER_DIR, d))])
NUM_GENDER_CLASSES = len(GENDER_CLASSES)
GENDER_MAP = {folder: idx for idx, folder in enumerate(GENDER_CLASSES)}

# ----------------- TRANSFORMS -----------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------- DATASET -----------------
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

def load_gender_dataset(gender_dir=GENDER_DIR, img_size=IMG_SIZE):
    X_gender, y_gender = [], []
    print("Loading gender data from cleaned directory...")
    for folder_name in GENDER_CLASSES:
        class_dir = os.path.join(gender_dir, folder_name)
        label = GENDER_MAP[folder_name]
        for fname in os.listdir(class_dir):
            if fname.endswith('.jpg'):
                img = Image.open(os.path.join(class_dir, fname)).convert('RGB')
                X_gender.append(img)  # Keep as PIL Image
                y_gender.append(label)
    return X_gender, y_gender  # <-- return lists of PIL Images


def split_gender_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_dataloader(X, y, batch_size=BATCH_SIZE, transforms=None):
    dataset = GenderDataset(X, y, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ----------------- MODEL -----------------
class ResNetGenderModel(nn.Module):
    def __init__(self, output_units=1):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze all layers except layer4 + FC
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # Add dropout to FC layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, output_units)
        )

    def forward(self, x):
        return self.resnet(x)

# ----------------- TRAIN -----------------
def train_gender_model(model, train_loader, val_loader, epochs=EPOCHS_GENDER, lr=1e-3, model_name='gender_model_resnet.pth'):
    model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            output = model(xb).squeeze()
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                output = model(xb).squeeze()
                loss = criterion(output, yb)
                val_loss += loss.item()
                pred = (torch.sigmoid(output) > 0.5).float()
                correct += (pred == yb).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_loader.dataset)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch} due to no improvement in Val Loss.")
            break

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

    # Save best model
    os.makedirs(MODEL_DIR, exist_ok=True)
    if best_model_state:
        torch.save(best_model_state, os.path.join(MODEL_DIR, model_name))
        print(f"Best model saved with Val Loss: {best_val_loss:.4f}")
    else:
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_name))

    # Plots
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Gender Training Loss')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'gender_loss_curve.png'))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Gender Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'gender_metric_curve.png'))
    plt.close()

# ----------------- EVALUATION -----------------
def evaluate_gender_model(model, test_loader):
    model.to(DEVICE)
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb).squeeze()
            pred = (torch.sigmoid(output) > 0.5).float()
            correct += (pred == yb).sum().item()
    acc = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {acc:.4f}")

# ----------------- MAIN -----------------
def main():
    print(f"Using device: {DEVICE}")

    X_gender, y_gender = load_gender_dataset()
    print(f"Total gender samples: {len(X_gender)}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_gender_dataset(X_gender, y_gender)

    train_loader = create_dataloader(X_train, y_train, transforms=train_transforms)
    val_loader = create_dataloader(X_val, y_val, transforms=val_transforms)
    test_loader = create_dataloader(X_test, y_test, transforms=val_transforms)

    model = ResNetGenderModel()
    train_gender_model(model, train_loader, val_loader, epochs=EPOCHS_GENDER)
    evaluate_gender_model(model, test_loader)
    print("Gender model trained, evaluated, and saved successfully!")

if __name__ == '__main__':
    main()
