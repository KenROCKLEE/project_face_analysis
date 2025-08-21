"""
Emotion Detection Training Script (FER2013 - Cleaned) in PyTorch
Author: Your Name
Date: 2025-08-20

This script trains a CNN model for emotion detection using the cleaned FER-2013 dataset.
Classes: angry, fearful, happy, neutral, sad, surprise
- Uses only 3000 images per class for balance
- Input size: 128x128
- Batch size: 32
- Saves the best model based on validation accuracy
- Incorporates ResNet-18 for transfer learning and fine-tuning
- Generates and saves training history plots
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import pandas as pd

# =========================
# Parameters
# =========================
DATASET_DIR = "datasets/clean/emotion"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SAMPLES_PER_CLASS = 3000
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 50
MODEL_DIR = 'models'
MODEL_SAVE_PATH = "emotion_model_resnet.pth"

# Get emotion classes
EMOTION_CLASSES = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
NUM_CLASSES = len(EMOTION_CLASSES)
EMOTION_MAP = {folder: idx for idx, folder in enumerate(EMOTION_CLASSES)}
print(f"Emotion classes: {EMOTION_CLASSES}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No GPU found, using CPU")

# =========================
# Dataset and DataLoader
# =========================

def get_balanced_dataset(dataset_dir, samples_per_class):
    filepaths, labels = [], []
    for emotion_class in EMOTION_CLASSES:
        class_dir = os.path.join(dataset_dir, emotion_class)
        files = os.listdir(class_dir)
        random.shuffle(files)
        selected = files[:samples_per_class]

        for f in selected:
            filepaths.append(os.path.join(class_dir, f))
            labels.append(EMOTION_MAP[emotion_class])

    return filepaths, labels

class EmotionDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# Data augmentation and normalization transforms
train_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and split dataset
filepaths, labels = get_balanced_dataset(DATASET_DIR, SAMPLES_PER_CLASS)
X_train, X_val, y_train, y_val = train_test_split(filepaths, labels, test_size=0.2, stratify=labels, random_state=42)

train_dataset = EmotionDataset(X_train, y_train, transform=train_transform)
val_dataset = EmotionDataset(X_val, y_val, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# =========================
# Model Definition (ResNet-18)
# =========================

class EmotionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load pre-trained ResNet-18 model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze all layers initially
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Replace the final classification layer for our task
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# =========================
# Training Function
# =========================

def train_model(model, train_loader, val_loader, epochs, lr, phase_name, model_dir, model_save_path):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5) 
    
    best_val_accuracy = 0.0
    patience_counter = 0

    train_losses, val_losses, val_accuracies = [], [], []

    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)
    full_model_path = os.path.join(model_dir, model_save_path)

    print(f"\n--- {phase_name} ---")
    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_preds += labels.size(0)
                correct_preds += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_preds / total_preds

        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        scheduler.step(avg_val_loss) # Or scheduler.step(val_accuracy) if monitoring accuracy

        # Check for improvement
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), full_model_path)
            print(f"Model saved! Val Accuracy improved to {best_val_accuracy:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch} due to no improvement in Val Accuracy.")
                break

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

    return train_losses, val_losses, val_accuracies

# =========================
# Plotting Function
# =========================

def plot_history(train_losses, val_losses, val_accuracies, phase_name, model_dir):
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title(f'{phase_name} - Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, f'{phase_name.replace(" ", "_").lower()}_loss.png'))
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_accuracies, 'g-o', label='Validation Accuracy')
    plt.title(f'{phase_name} - Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, f'{phase_name.replace(" ", "_").lower()}_accuracy.png'))
    plt.close()

# =========================
# Main Training Logic
# =========================

# Phase 1: Feature Extraction
model = EmotionModel(NUM_CLASSES)
print("\nPhase 1: Training only the top layers (Feature Extraction)")
# The final fc layer is the only one with requires_grad=True by default
# so we only need to pass its parameters to the optimizer
optimizer = optim.Adam(model.resnet.fc.parameters(), lr=1e-3)
phase1_train_losses, phase1_val_losses, phase1_val_accuracies = train_model(
    model, train_loader, val_loader, EPOCHS_PHASE1, lr=1e-3, 
    phase_name="Phase 1: Feature Extraction", model_dir=MODEL_DIR, model_save_path=MODEL_SAVE_PATH
)
# Plot graphs for Phase 1
plot_history(phase1_train_losses, phase1_val_losses, phase1_val_accuracies, 
             "Phase 1: Feature Extraction", MODEL_DIR)


# Phase 2: Fine-tuning
print("\nPhase 2: Fine-tuning the entire model")
# Load the best model from Phase 1
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, MODEL_SAVE_PATH)))

# Unfreeze the last few blocks for fine-tuning
# A common strategy is to unfreeze `layer4` and the `fc` layer for ResNet
for param in model.resnet.parameters():
    param.requires_grad = True # Unfreeze all layers of ResNet for fine-tuning

# Recompile the optimizer with all parameters and a much lower learning rate
optimizer_ft = optim.Adam(model.parameters(), lr=1e-5) 
phase2_train_losses, phase2_val_losses, phase2_val_accuracies = train_model(
    model, train_loader, val_loader, EPOCHS_PHASE2, lr=1e-5, 
    phase_name="Phase 2: Fine-tuning", model_dir=MODEL_DIR, model_save_path=MODEL_SAVE_PATH
)
# Plot graphs for Phase 2
plot_history(phase2_train_losses, phase2_val_losses, phase2_val_accuracies, 
             "Phase 2: Fine-tuning", MODEL_DIR)


print(f"✅ Training complete. Best model saved to {os.path.join(MODEL_DIR, MODEL_SAVE_PATH)}")

