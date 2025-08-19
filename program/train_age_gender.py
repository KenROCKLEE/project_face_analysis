"""
UTKFace Age and Gender Prediction Pipeline in PyTorch
Author: Your Name
Date: 2025-08-18

This script loads the UTKFace dataset, preprocesses images,
builds and trains two separate models for age (regression) and gender (binary classification),
evaluates them, and saves models and training curves.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# ----------------- CONFIG -----------------
IMG_SIZE = (128, 128)
AGE_DIR = 'datasets/raw/UTKFace'  # Directory for age data (raw)
GENDER_DIR = 'datasets/clean/gender' # Directory for gender data (cleaned)
MODEL_DIR = 'models'
BATCH_SIZE = 32
EPOCHS_AGE = 100 # Increased epochs for age model
EPOCHS_GENDER = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gender classes
GENDER_CLASSES = sorted([d for d in os.listdir(GENDER_DIR) if os.path.isdir(os.path.join(GENDER_DIR,d))])
NUM_GENDER_CLASSES = len(GENDER_CLASSES)

# Data augmentation for age model
age_train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

age_val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# ----------------- UTILS -----------------
def load_dataset(age_dir=AGE_DIR, gender_dir=GENDER_DIR, img_size=IMG_SIZE):
    X_age, y_age = [], []
    X_gender, y_gender = [], []

    # Load age data from the raw UTKFace directory
    print("Loading age data from raw directory...")
    for fname in os.listdir(age_dir):
        if fname.endswith('.jpg'):
            try:
                # Parse age from filename
                parts = fname.split('_')
                age = int(parts[0])
                
                img_path = os.path.join(age_dir, fname)
                img = Image.open(img_path).convert('RGB').resize(img_size)
                img = np.array(img, dtype=np.float32) / 255.0
                
                X_age.append(img)
                y_age.append(age)
            
            except (ValueError, IndexError):
                # Skip files that don't match the expected naming format
                continue
    
    # Load gender data from the cleaned gender directory
    print("Loading gender data from cleaned directory...")
    gender_map = {folder_name: idx for idx, folder_name in enumerate(GENDER_CLASSES)}
    for folder_name in GENDER_CLASSES:
        class_dir = os.path.join(gender_dir, folder_name)
        gender_label = gender_map[folder_name]
        for fname in os.listdir(class_dir):
            if fname.endswith('.jpg'):
                img_path = os.path.join(class_dir, fname)
                img = Image.open(img_path).convert('RGB').resize(img_size)
                img = np.array(img, dtype=np.float32) / 255.0
                X_gender.append(img)
                y_gender.append(gender_label)
    
    return np.array(X_age), np.array(y_age), np.array(X_gender), np.array(y_gender)

def split_dataset(X_age, y_age, X_gender, y_gender, test_size=0.2, val_size=0.1, random_state=42):
    # Age split
    X_age_train, X_age_test, y_age_train, y_age_test = train_test_split(X_age, y_age, test_size=test_size, random_state=random_state)
    X_age_train, X_age_val, y_age_train, y_age_val = train_test_split(X_age_train, y_age_train, test_size=val_size, random_state=random_state)

    # Gender split (untouched)
    X_gender_train, X_gender_test, y_gender_train, y_gender_test = train_test_split(X_gender, y_gender, test_size=test_size, random_state=random_state)
    X_gender_train, X_gender_val, y_gender_train, y_gender_val = train_test_split(X_gender_train, y_gender_train, test_size=val_size, random_state=random_state)

    return X_age_train, X_age_val, X_age_test, y_age_train, y_age_val, y_age_test, \
           X_gender_train, X_gender_val, X_gender_test, y_gender_train, y_gender_val, y_gender_test

class AgeDataset(torch.utils.data.Dataset):
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
        return image, torch.tensor(label, dtype=torch.float32).unsqueeze(0)

def create_dataloader(X, y, batch_size=BATCH_SIZE, task='age', transforms=None):
    if task == 'age':
        dataset = AgeDataset(X, y, transform=transforms)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else: # gender
        X_tensor = torch.tensor(X, dtype=torch.float32).permute(0,3,1,2)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ----------------- MODEL -----------------
class BaseCNN(nn.Module):
    def __init__(self, output_units=1, task='age'):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(128),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*(IMG_SIZE[0]//8)*(IMG_SIZE[1]//8), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_units)
        )
        self.task = task

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if self.task=='gender':
            x = torch.sigmoid(x)
        return x

# ----------------- TRAIN -----------------
def train_model(model, train_loader, val_loader, task='age', epochs=20, lr=1e-3, model_name='model.pth'):
    model.to(DEVICE)
    if task=='age':
        criterion = nn.L1Loss() # MAE for regression
    else:  # gender
        criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Early Stopping and Learning Rate Scheduler for age model
    if task == 'age':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        best_val_mae = float('inf')
        patience_counter = 0
        best_model_state = None

    train_losses, val_losses, val_metrics = [], [], []

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb)
            if task=='age':
                loss = criterion(output, yb)
            else: # gender
                output = output.squeeze()
                yb = yb.float()
                loss = criterion(output, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_metric = 0 
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                output = model(xb)
                if task=='age':
                    loss = criterion(output, yb)
                    val_metric += torch.abs(output - yb).mean().item()
                else: # gender
                    output = output.squeeze()
                    yb = yb.float()
                    loss = criterion(output, yb)
                    pred = (output>0.5).float()
                    val_metric += (pred==yb).sum().item()
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        if task=='age':
            val_metric = val_metric / len(val_loader)
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val MAE: {val_metric:.4f}")

            # Check for early stopping and save best model
            scheduler.step(val_metric)
            if val_metric < best_val_mae:
                best_val_mae = val_metric
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch} due to no improvement in Val MAE.")
                break

        else:
            val_metric = val_metric / len(val_loader.dataset)
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {val_metric:.4f}")

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        val_metrics.append(val_metric)

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    if task == 'age' and best_model_state:
        torch.save(best_model_state, os.path.join(MODEL_DIR, model_name))
        print(f"Age model saved from best epoch with Val MAE: {best_val_mae:.4f}")
    else:
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_name))

    # Plot curves
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{task} Training Loss')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, f'{task}_loss_curve.png'))
    plt.close()

    plt.figure(figsize=(8,4))
    metric_label = 'Validation MAE' if task=='age' else 'Validation Accuracy'
    plt.plot(val_metrics, label=metric_label)
    plt.title(f'{task} Validation Metric')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, f'{task}_metric_curve.png'))
    plt.close()

# ----------------- MAIN -----------------
def main():
    # Check for GPU
    if torch.cuda.is_available():
        print(f"✅ GPU is available and will be used! Device: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ GPU not available, using CPU. This will be slow.")

    print("Loading dataset...")
    X_age, y_age, X_gender, y_gender = load_dataset()
    print(f"Total age samples: {len(X_age)} | Total gender samples: {len(X_gender)}")

    # Split
    X_age_train, X_age_val, X_age_test, y_age_train, y_age_val, y_age_test, \
    X_gender_train, X_gender_val, X_gender_test, y_gender_train, y_gender_val, y_gender_test = \
        split_dataset(X_age, y_age, X_gender, y_gender)

    # Dataloaders
    age_train_loader = create_dataloader(X_age_train, y_age_train, task='age', transforms=age_train_transforms)
    age_val_loader = create_dataloader(X_age_val, y_age_val, task='age', transforms=age_val_transforms)
    gender_train_loader = create_dataloader(X_gender_train, y_gender_train, task='gender')
    gender_val_loader = create_dataloader(X_gender_val, y_gender_val, task='gender')

    # Train Age model
    print("Training Age model...")
    age_model = BaseCNN(output_units=1, task='age')
    train_model(age_model, age_train_loader, age_val_loader, task='age', epochs=EPOCHS_AGE, model_name='age_model.pth')

    # Train Gender model
    print("Training Gender model...")
    gender_model = BaseCNN(output_units=1, task='gender')
    train_model(gender_model, gender_train_loader, gender_val_loader, task='gender', epochs=EPOCHS_GENDER, model_name='gender_model.pth')

    print("All models trained and saved successfully!")

if __name__ == '__main__':
    main()