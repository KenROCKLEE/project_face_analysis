"""
UTKFace Age Classification with ResNet-18
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
from sklearn.utils.class_weight import compute_class_weight

# ----------------- CONFIG -----------------
IMG_SIZE = (128, 128)
AGE_DIR = 'datasets/clean/age'  # preprocessed age bins
MODEL_DIR = 'models'
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10  # Train only the head
EPOCHS_PHASE2 = 40  # Fine-tune the full model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATIENCE = 15  # early stopping patience

# Age groups
AGE_GROUPS = [
    (0, 2),
    (3, 6),
    (7, 12),
    (13, 19),
    (20, 29),
    (30, 39),
    (40, 49),
    (50, 59),
    (60, 114)
]
AGE_CLASS_LABELS = [f"{start}-{end}" for start, end in AGE_GROUPS]
NUM_AGE_CLASSES = len(AGE_CLASS_LABELS)

# Normalization
normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

# Transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    normalize_transform
])

val_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    normalize_transform
])

# ----------------- DATASET -----------------
class AgeDatasetClassification(Dataset):
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

def load_age_dataset(age_dir=AGE_DIR, img_size=IMG_SIZE):
    X_age, y_age = [], []
    print("Loading age data from cleaned directories...")

    for class_idx, folder_name in enumerate(AGE_CLASS_LABELS):
        class_dir = os.path.join(age_dir, folder_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.endswith('.jpg'):
                img_path = os.path.join(class_dir, fname)
                img = Image.open(img_path).convert('RGB').resize(img_size)
                X_age.append(img)  
                y_age.append(class_idx)
    return X_age, np.array(y_age)  

def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_dataloader(X, y, batch_size=BATCH_SIZE, transforms=None):
    dataset = AgeDatasetClassification(X, y, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ----------------- MODEL -----------------
class ResNetAgeClassification(nn.Module):
    def __init__(self, num_classes=NUM_AGE_CLASSES):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# ----------------- TRAIN -----------------
def train_age_model(model, train_loader, val_loader, y_train, model_name='age_model_resnet.pth'):
    model.to(DEVICE)
    
    # Calculate class weights for the imbalanced dataset
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses, val_losses, val_accuracies = [], [], []
    
    # --- PHASE 1: Train the classifier head only ---
    print("\n--- Starting Phase 1: Training classifier head ---")
    for param in model.resnet.parameters():
        param.requires_grad = False
    for param in model.resnet.fc.parameters():
        param.requires_grad = True

    optimizer_phase1 = optim.Adam(model.resnet.fc.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, EPOCHS_PHASE1 + 1):
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb)
            loss = criterion(output, yb)
            optimizer_phase1.zero_grad()
            loss.backward()
            optimizer_phase1.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        avg_val_loss, val_acc = validate_model(model, val_loader, criterion)
        print(f"Phase 1 - Epoch {epoch}/{EPOCHS_PHASE1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # --- PHASE 2: Fine-tune the full model ---
    print("\n--- Starting Phase 2: Fine-tuning ResNet layers ---")
    for param in model.resnet.layer3.parameters():
        param.requires_grad = True
    for param in model.resnet.layer4.parameters():
        param.requires_grad = True

    # Layer-specific learning rates and weight decay for fine-tuning
    optimizer_phase2 = optim.Adam([
        {'params': model.resnet.layer3.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4}, # Lower learning rate
        {'params': model.resnet.layer4.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4}, # Lower learning rate
        {'params': model.resnet.fc.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4}     # Lower learning rate
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_phase2, mode='min', factor=0.5, patience=5)
    patience_counter = 0

    for epoch in range(1, EPOCHS_PHASE2 + 1):
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb)
            loss = criterion(output, yb)
            optimizer_phase2.zero_grad()
            loss.backward()
            optimizer_phase2.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        avg_val_loss, val_acc = validate_model(model, val_loader, criterion)
        print(f"Phase 2 - Epoch {epoch}/{EPOCHS_PHASE2} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

    os.makedirs(MODEL_DIR, exist_ok=True)
    if best_model_state:
        torch.save(best_model_state, os.path.join(MODEL_DIR, model_name))
        print(f"\nBest model saved with Val Loss: {best_val_loss:.4f}")

    # Plots
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Age Training Loss')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'age_loss_curve.png'))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Age Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'age_metric_curve.png'))
    plt.close()

def validate_model(model, loader, criterion):
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb)
            loss = criterion(output, yb)
            pred = output.argmax(dim=1)
            val_correct += (pred == yb).sum().item()
            val_loss += loss.item()
    avg_val_loss = val_loss / len(loader)
    val_acc = val_correct / len(loader.dataset)
    return avg_val_loss, val_acc

# ----------------- EVALUATION -----------------
def evaluate_age_model(model, test_loader):
    model.to(DEVICE)
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb)
            pred = output.argmax(dim=1)
            correct += (pred == yb).sum().item()
    acc = correct / len(test_loader.dataset)
    print(f"\nTest Accuracy: {acc:.4f}")

# ----------------- MAIN -----------------
def main():
    if torch.cuda.is_available():
        print(f"✅ GPU is available and will be used! Device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU found, using CPU")

    X_age, y_age = load_age_dataset()
    print(f"Total samples: {len(X_age)}")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X_age, y_age)

    train_loader = create_dataloader(X_train, y_train, transforms=train_transforms)
    val_loader = create_dataloader(X_val, y_val, transforms=val_transforms)
    test_loader = create_dataloader(X_test, y_test, transforms=val_transforms)

    model = ResNetAgeClassification()
    train_age_model(model, train_loader, val_loader, y_train)
    
    best_model_path = os.path.join(MODEL_DIR, 'age_model_resnet.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model for final evaluation.")
        evaluate_age_model(model, test_loader)
    else:
        print("Best model not found. Cannot perform final evaluation.")

    print("\nAge classification model trained, evaluated, and saved!")

if __name__ == '__main__':
    main()