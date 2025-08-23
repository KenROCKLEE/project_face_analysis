"""
UTKFace Age Classification with ResNet-18 (Top-2 Accuracy)
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
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 70
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATIENCE = 15  # early stopping patience

# Age groups
AGE_GROUPS = [
    (0, 2), (3, 6), (7, 12), (13, 19),
    (20, 29), (30, 39), (40, 49), (50, 59), (60, 114)
]
AGE_CLASS_LABELS = [f"{start}-{end}" for start, end in AGE_GROUPS]
NUM_AGE_CLASSES = len(AGE_CLASS_LABELS)

# Normalization
normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])

# ----------------- TRANSFORMS -----------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
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
            nn.Dropout(0.6),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# ----------------- TRAIN & VALIDATION -----------------
def validate_model(model, loader, criterion):
    model.eval()
    val_loss = 0
    val_correct_top1 = 0
    val_correct_top2 = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb)
            loss = criterion(output, yb)
            
            # Top-1
            pred_top1 = output.argmax(dim=1)
            val_correct_top1 += (pred_top1 == yb).sum().item()
            
            # Top-2
            top2_vals, top2_idx = output.topk(2, dim=1)
            val_correct_top2 += sum([yb[i] in top2_idx[i] for i in range(len(yb))])
            
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(loader)
    val_acc_top1 = val_correct_top1 / len(loader.dataset)
    val_acc_top2 = val_correct_top2 / len(loader.dataset)
    return avg_val_loss, val_acc_top1, val_acc_top2

def train_age_model(model, train_loader, val_loader, y_train, model_name='age_model_resnet.pth'):
    model.to(DEVICE)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_val_loss = float('inf')
    best_model_state = None
    train_losses, val_losses, val_acc_top1_list, val_acc_top2_list = [], [], [], []

    # --- PHASE 1 ---
    print("\n--- Phase 1: Training classifier head ---")
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
        avg_val_loss, val_acc_top1, val_acc_top2 = validate_model(model, val_loader, criterion)
        print(f"Phase1-Epoch {epoch}/{EPOCHS_PHASE1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Top1: {val_acc_top1:.4f} | Top2: {val_acc_top2:.4f}")

    # --- PHASE 2 ---
    print("\n--- Phase 2: Fine-tuning full model ---")
    for param in model.resnet.layer3.parameters():
        param.requires_grad = True
    for param in model.resnet.layer4.parameters():
        param.requires_grad = True

    optimizer_phase2 = optim.Adam([
        {'params': model.resnet.layer3.parameters(), 'lr': 1e-6, 'weight_decay': 1e-4},
        {'params': model.resnet.layer4.parameters(), 'lr': 1e-6, 'weight_decay': 1e-4},
        {'params': model.resnet.fc.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4}
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
        avg_val_loss, val_acc_top1, val_acc_top2 = validate_model(model, val_loader, criterion)
        print(f"Phase2-Epoch {epoch}/{EPOCHS_PHASE2} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Top1: {val_acc_top1:.4f} | Top2: {val_acc_top2:.4f}")

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_acc_top1_list.append(val_acc_top1)
        val_acc_top2_list.append(val_acc_top2)

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
    plt.plot(val_acc_top1_list, label='Val Top-1 Acc')
    plt.plot(val_acc_top2_list, label='Val Top-2 Acc')
    plt.title('Age Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'age_metric_curve.png'))
    plt.close()

# ----------------- TEST -----------------
def evaluate_age_model(model, test_loader):
    model.to(DEVICE)
    model.eval()
    correct_top1, correct_top2 = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb)
            
            pred_top1 = output.argmax(dim=1)
            correct_top1 += (pred_top1 == yb).sum().item()
            
            top2_vals, top2_idx = output.topk(2, dim=1)
            correct_top2 += sum([yb[i] in top2_idx[i] for i in range(len(yb))])
    
    acc_top1 = correct_top1 / len(test_loader.dataset)
    acc_top2 = correct_top2 / len(test_loader.dataset)
    print(f"\nTest Accuracy Top-1: {acc_top1:.4f}")
    print(f"Test Accuracy Top-2: {acc_top2:.4f}")

# ----------------- MAIN -----------------
def main():
    print(f"Using device: {DEVICE}")
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

if __name__ == '__main__':
    main()
