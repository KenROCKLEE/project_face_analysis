"""
Emotion Prediction Pipeline in PyTorch
Author: Your Name
Date: 2025-08-22
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image

# ----------------- CONFIG -----------------
IMG_SIZE = (128, 128)
EMOTION_DIR = 'datasets/clean/emotion'
MODEL_DIR = 'models'
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Emotion classes and mapping
EMOTION_CLASSES = sorted([d for d in os.listdir(EMOTION_DIR) if os.path.isdir(os.path.join(EMOTION_DIR, d))])
NUM_EMOTION_CLASSES = len(EMOTION_CLASSES)
EMOTION_MAP = {folder: idx for idx, folder in enumerate(EMOTION_CLASSES)}

# Data normalization
normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Transforms for training (with data augmentation) and validation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_transform
])

val_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    normalize_transform
])

# ----------------- UTILS -----------------
def load_emotion_dataset(emotion_dir=EMOTION_DIR, img_size=IMG_SIZE):
    X_emotion, y_emotion = [], []
    print("Loading emotion data from cleaned directory...")
    
    for folder_name in EMOTION_CLASSES:
        class_dir = os.path.join(emotion_dir, folder_name)
        emotion_label = EMOTION_MAP[folder_name]
        for fname in os.listdir(class_dir):
            if fname.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(class_dir, fname)
                img = Image.open(img_path).convert('RGB').resize(img_size)
                X_emotion.append(img)
                y_emotion.append(emotion_label)

    return np.array(X_emotion), np.array(y_emotion)

def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test

class EmotionDataset(Dataset):
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
        return image, torch.tensor(label, dtype=torch.long)

def create_dataloader(X, y, batch_size=BATCH_SIZE, transforms=None, sampler=None):
    dataset = EmotionDataset(X, y, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler)

def get_weighted_sampler(y_train):
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler

# ----------------- MODEL -----------------
class ResNetEmotionModel(nn.Module):
    def __init__(self, output_units=NUM_EMOTION_CLASSES):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Phase 1: Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Replace the final classification layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, output_units)
        
        self.freeze_until = "fc"

    def unfreeze_layers(self, phase_name):
        if phase_name == "fine-tuning":
            # Phase 2: Unfreeze more layers for fine-tuning
            for name, param in self.resnet.named_parameters():
                if "layer4" in name or "fc" in name:
                    param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)

# ----------------- TRAIN -----------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=1e-3, model_name='emotion_model.pth'):
    model.to(DEVICE)
    # For multi-class classification, use CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_accuracy = 0
    patience_counter = 0
    best_model_state = None

    train_losses, val_losses, val_accuracies = [], [], []

    print("--- Phase 1: Feature Extraction ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0
        correct_train = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb)
            loss = criterion(output, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(output, 1)
            correct_train += (preds == yb).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        correct_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                output = model(xb)
                loss = criterion(output, yb)
                val_loss += loss.item()
                _, preds = torch.max(output, 1)
                correct_val += (preds == yb).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = correct_val / len(val_loader.dataset)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {avg_val_accuracy:.4f}")
        
        scheduler.step(avg_val_loss)

        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict()
            print(f"Model saved! Val Accuracy improved to {best_val_accuracy:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= 10:
            print(f"Early stopping at epoch {epoch} due to no improvement in Val Accuracy.")
            break
            
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

    os.makedirs(MODEL_DIR, exist_ok=True)
    if best_model_state:
        torch.save(best_model_state, os.path.join(MODEL_DIR, model_name))
        print(f"✅ Training complete. Best model saved to {os.path.join(MODEL_DIR, model_name)}")
    else:
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_name))
    
    # Save plots
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Emotion Training Loss')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'emotion_loss_curve.png'))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Emotion Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'emotion_metric_curve.png'))
    plt.close()

# ----------------- EVALUATION -----------------
def evaluate_model(model, test_loader):
    model.to(DEVICE)
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb)
            _, preds = torch.max(output, 1)
            correct += (preds == yb).sum().item()
    
    test_accuracy = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {test_accuracy:.4f}")

# ----------------- MAIN -----------------
def main():
    if torch.cuda.is_available():
        print(f"✅ GPU is available and will be used! Device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU found, using CPU")

    print("Loading emotion dataset...")
    X_emotion, y_emotion = load_emotion_dataset()
    print(f"Total emotion samples: {len(X_emotion)}")
    
    # Print class distribution
    unique_classes, counts = np.unique(y_emotion, return_counts=True)
    print("Class Distribution:", dict(zip(unique_classes, counts)))
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X_emotion, y_emotion)
    
    # Weighted Random Sampler for balanced training
    sampler = get_weighted_sampler(y_train)
    
    # Dataloaders
    train_loader = create_dataloader(X_train, y_train, transforms=train_transforms, sampler=sampler)
    val_loader = create_dataloader(X_val, y_val, transforms=val_transforms)
    test_loader = create_dataloader(X_test, y_test, transforms=val_transforms)
    
    # Train Emotion model
    print("\nTraining Emotion model with ResNet...")
    emotion_model = ResNetEmotionModel(output_units=NUM_EMOTION_CLASSES)
    train_model(emotion_model, train_loader, val_loader, epochs=EPOCHS, lr=1e-3, model_name='emotion_model_resnet.pth')
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    best_model = ResNetEmotionModel(output_units=NUM_EMOTION_CLASSES)
    best_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'emotion_model_resnet.pth')))
    evaluate_model(best_model, test_loader)
    
    print("All models trained, evaluated, and saved successfully!")

if __name__ == '__main__':
    main()