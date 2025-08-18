"""
UTKFace Age and Gender Prediction Pipeline in PyTorch
Author: Your Name
Date: 2025-08-18

This script loads the UTKFace dataset, preprocesses images, builds and trains two separate models 
for age (regression) and gender (binary classification),
evaluates them with metrics, and saves the results and models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# Constants
IMG_SIZE = (128, 128)
DATASET_DIR = 'datasets/raw/UTKFace'
MODEL_DIR = 'models'
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load dataset
def load_dataset(dataset_dir=DATASET_DIR, img_size=IMG_SIZE, max_images=None):
    from PIL import Image
    X, y_age, y_gender = [], [], []
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
    if max_images:
        files = files[:max_images]
    for fname in files:
        try:
            age, gender, _, _ = fname.split('_', 3)
            img_path = os.path.join(dataset_dir, fname)
            img = Image.open(img_path).convert('RGB').resize(img_size)
            img = np.array(img, dtype=np.float32) / 255.0
            X.append(img)
            y_age.append(float(age))
            y_gender.append(int(gender))
        except Exception as e:
            print(f"Skipping {fname}: {e}")
    X = np.array(X)
    y_age = np.array(y_age)
    y_gender = np.array(y_gender)
    return X, y_age, y_gender

# 2. Split dataset
def split_dataset(X, y_age, y_gender, test_size=0.2, val_size=0.1, random_state=42):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
        X, y_age, y_gender, test_size=test_size, random_state=random_state)
    X_train, X_val, y_age_train, y_age_val, y_gender_train, y_gender_val = train_test_split(
        X_train, y_age_train, y_gender_train, test_size=val_size, random_state=random_state)
    return (X_train, X_val, X_test, y_age_train, y_age_val, y_age_test,
            y_gender_train, y_gender_val, y_gender_test)

# 3. PyTorch dataset loader
def create_dataloader(X, y, batch_size=BATCH_SIZE, task='age'):
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0,3,1,2)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# 4. CNN model
class BaseCNN(nn.Module):
    def __init__(self, output_units=1, task='age'):
        super().__init__()
        self.task = task
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(128),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * (IMG_SIZE[0]//8) * (IMG_SIZE[1]//8), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_units)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if self.task=='gender':
            x = torch.sigmoid(x)
        return x

# 5. Training function with metrics
def train_model(model, train_loader, val_loader, task='age', epochs=EPOCHS, lr=1e-3, model_name='model.pth'):
    model.to(DEVICE)
    if task=='age':
        criterion = nn.MSELoss()
    elif task=='gender':
        criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses, val_metrics = [], [], []

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb).squeeze()
            loss = criterion(output, yb)
            loss.backward()
            optimizer.zero_grad()
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
                output = model(xb).squeeze()
                loss = criterion(output, yb)
                val_loss += loss.item()

                # Metrics
                if task=='gender':
                    pred = (output>0.5).float()
                    correct += (pred==yb).sum().item()
                    total += yb.size(0)
                elif task=='age':
                    # MAE
                    correct += torch.sum(torch.abs(output-yb)).item()
                    total += yb.size(0)

        avg_val = val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        # Compute metric
        if task=='age':
            metric_val = correct / total  # mean absolute error
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val MAE: {metric_val:.2f}")
        else:
            metric_val = correct / total
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {metric_val:.4f}")

        val_metrics.append(metric_val)

    # Save model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_name))

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{task} Training Loss')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, f'{task}_loss_curve.png'))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(val_metrics, label='Validation Metric')
    plt.title(f'{task} Validation Metric')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, f'{task}_metric_curve.png'))
    plt.close()

# 6. Main pipeline
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Loading dataset...")
    X, y_age, y_gender = load_dataset()
    print(f"Total samples: {len(X)}")
    
    # Split
    X_train, X_val, X_test, y_age_train, y_age_val, y_age_test, \
    y_gender_train, y_gender_val, y_gender_test = split_dataset(X, y_age, y_gender)

    # Dataloaders
    age_train_loader = create_dataloader(X_train, y_age_train, task='age')
    age_val_loader = create_dataloader(X_val, y_age_val, task='age')
    gender_train_loader = create_dataloader(X_train, y_gender_train, task='gender')
    gender_val_loader = create_dataloader(X_val, y_gender_val, task='gender')

    # Age
    print("Training Age model...")
    age_model = BaseCNN(output_units=1, task='age')
    train_model(age_model, age_train_loader, age_val_loader, task='age', model_name='age_model.pth')

    # Gender
    print("Training Gender model...")
    gender_model = BaseCNN(output_units=1, task='gender')
    train_model(gender_model, gender_train_loader, gender_val_loader, task='gender', model_name='gender_model.pth')

    print("All models trained and saved.")

if __name__ == '__main__':
    main()
