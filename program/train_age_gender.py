"""
UTKFace Age and Gender Prediction Pipeline in PyTorch
Author: Your Name
Date: 2025-08-18

This script loads the UTKFace dataset (cleaned version), preprocesses images,
builds and trains two separate models for age (classification) and gender (binary classification),
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
DATASET_DIR = 'datasets/clean'  # use clean folder
AGE_DIR = os.path.join(DATASET_DIR, 'age')
GENDER_DIR = os.path.join(DATASET_DIR, 'gender')
MODEL_DIR = 'models'
BATCH_SIZE = 32
EPOCHS_AGE = 60
EPOCHS_GENDER = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Age class labels (folder names)
AGE_CLASSES = sorted([d for d in os.listdir(AGE_DIR) if os.path.isdir(os.path.join(AGE_DIR,d))])
NUM_AGE_CLASSES = len(AGE_CLASSES)

# Gender classes
GENDER_CLASSES = sorted([d for d in os.listdir(GENDER_DIR) if os.path.isdir(os.path.join(GENDER_DIR,d))])
NUM_GENDER_CLASSES = len(GENDER_CLASSES)

# ----------------- UTILS -----------------
def load_dataset(clean_dir=DATASET_DIR, img_size=IMG_SIZE):
    X_age, y_age = [], []
    X_gender, y_gender = [], []

    # Load age data
    for idx, age_class in enumerate(AGE_CLASSES):
        class_dir = os.path.join(AGE_DIR, age_class)
        for fname in os.listdir(class_dir):
            if fname.endswith('.jpg'):
                img_path = os.path.join(class_dir, fname)
                img = Image.open(img_path).convert('RGB').resize(img_size)
                img = np.array(img, dtype=np.float32) / 255.0
                X_age.append(img)
                y_age.append(idx)
    
    # Load gender data
    for idx, gender_class in enumerate(GENDER_CLASSES):
        class_dir = os.path.join(GENDER_DIR, gender_class)
        for fname in os.listdir(class_dir):
            if fname.endswith('.jpg'):
                img_path = os.path.join(class_dir, fname)
                img = Image.open(img_path).convert('RGB').resize(img_size)
                img = np.array(img, dtype=np.float32) / 255.0
                X_gender.append(img)
                y_gender.append(idx)

    return np.array(X_age), np.array(y_age), np.array(X_gender), np.array(y_gender)

def split_dataset(X_age, y_age, X_gender, y_gender, test_size=0.2, val_size=0.1, random_state=42):
    # Age split
    X_age_train, X_age_test, y_age_train, y_age_test = train_test_split(X_age, y_age, test_size=test_size, random_state=random_state)
    X_age_train, X_age_val, y_age_train, y_age_val = train_test_split(X_age_train, y_age_train, test_size=val_size, random_state=random_state)

    # Gender split
    X_gender_train, X_gender_test, y_gender_train, y_gender_test = train_test_split(X_gender, y_gender, test_size=test_size, random_state=random_state)
    X_gender_train, X_gender_val, y_gender_train, y_gender_val = train_test_split(X_gender_train, y_gender_train, test_size=val_size, random_state=random_state)

    return X_age_train, X_age_val, X_age_test, y_age_train, y_age_val, y_age_test, \
           X_gender_train, X_gender_val, X_gender_test, y_gender_train, y_gender_val, y_gender_test

def create_dataloader(X, y, batch_size=BATCH_SIZE):
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0,3,1,2)
    y_tensor = torch.tensor(y, dtype=torch.long)  # use long for classification
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
        criterion = nn.CrossEntropyLoss()
    else:  # gender
        criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, val_metrics = [], [], []

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb)
            if task=='gender':
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
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                output = model(xb)
                if task=='gender':
                    output = output.squeeze()
                    yb = yb.float()
                loss = criterion(output, yb)
                val_loss += loss.item()

                # Metrics
                if task=='age':
                    pred = output.argmax(dim=1)
                    correct += (pred==yb).sum().item()
                    total += yb.size(0)
                else:
                    pred = (output>0.5).float()
                    correct += (pred==yb).sum().item()
                    total += yb.size(0)

        avg_val = val_loss / len(val_loader)
        val_metric = correct / total
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        val_metrics.append(val_metric)

        if task=='age':
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {val_metric:.4f}")
        else:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {val_metric:.4f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
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
    plt.plot(val_metrics, label='Validation Accuracy')
    plt.title(f'{task} Validation Metric')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, f'{task}_metric_curve.png'))
    plt.close()

# ----------------- MAIN -----------------
def main():
    print("Loading dataset...")
    X_age, y_age, X_gender, y_gender = load_dataset()
    print(f"Total age samples: {len(X_age)} | Total gender samples: {len(X_gender)}")

    # Split
    X_age_train, X_age_val, X_age_test, y_age_train, y_age_val, y_age_test, \
    X_gender_train, X_gender_val, X_gender_test, y_gender_train, y_gender_val, y_gender_test = \
        split_dataset(X_age, y_age, X_gender, y_gender)

    # Dataloaders
    age_train_loader = create_dataloader(X_age_train, y_age_train)
    age_val_loader = create_dataloader(X_age_val, y_age_val)
    gender_train_loader = create_dataloader(X_gender_train, y_gender_train)
    gender_val_loader = create_dataloader(X_gender_val, y_gender_val)

    # Train Age model
    print("Training Age model...")
    age_model = BaseCNN(output_units=NUM_AGE_CLASSES, task='age')
    train_model(age_model, age_train_loader, age_val_loader, task='age', epochs=EPOCHS_AGE, model_name='age_model.pth')

    # Train Gender model
    print("Training Gender model...")
    gender_model = BaseCNN(output_units=1, task='gender')
    train_model(gender_model, gender_train_loader, gender_val_loader, task='gender', epochs=EPOCHS_GENDER, model_name='gender_model.pth')

    print("All models trained and saved successfully!")

if __name__ == '__main__':
    main()
