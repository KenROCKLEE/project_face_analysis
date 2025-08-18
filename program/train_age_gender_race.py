"""
UTKFace Age, Gender, Race Prediction Pipeline (PyTorch)
Author: Your Name
Date: 2025-08-18
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ============================
# Constants
# ============================
IMG_SIZE = 128
DATASET_DIR = 'datasets/raw/UTKFace'
MODEL_DIR = 'models'
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_RACES = 5

# ============================
# Dataset
# ============================
class UTKFaceDataset(Dataset):
    def __init__(self, image_paths, ages, genders, races, transform=None):
        self.image_paths = image_paths
        self.ages = ages
        self.genders = genders
        self.races = races
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        age = torch.tensor(self.ages[idx], dtype=torch.float32)
        gender = torch.tensor(self.genders[idx], dtype=torch.float32)
        race = torch.tensor(self.races[idx], dtype=torch.long)
        return img, age, gender, race

# ============================
# Load dataset
# ============================
def load_dataset(dataset_dir=DATASET_DIR, max_images=None):
    image_paths, y_age, y_gender, y_race = [], [], [], []
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
    if max_images:
        files = files[:max_images]
    for f in files:
        try:
            age, gender, race, _ = f.split("_", 3)
            image_paths.append(os.path.join(dataset_dir, f))
            y_age.append(int(age))
            y_gender.append(int(gender))
            y_race.append(int(race))
        except:
            continue
    return image_paths, y_age, y_gender, y_race

# ============================
# Model
# ============================
class CNNModel(nn.Module):
    def __init__(self, output_units, task_type='regression'):
        super().__init__()
        self.task_type = task_type
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*(IMG_SIZE//8)*(IMG_SIZE//8), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_units)
        )
        if self.task_type == 'classification' and output_units == 1:
            self.activation = nn.Sigmoid()
        elif self.task_type == 'classification':
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.activation(x)
        return x

# ============================
# Training function
# ============================
def train_model(model, dataloader, criterion, optimizer, task):
    model.train()
    running_loss = 0.0
    for imgs, ages, genders, races in dataloader:
        imgs = imgs.to(DEVICE)
        if task == 'age':
            labels = ages.unsqueeze(1).to(DEVICE)
        elif task == 'gender':
            labels = genders.unsqueeze(1).to(DEVICE)
        elif task == 'race':
            labels = races.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(dataloader.dataset)

# ============================
# Evaluation function
# ============================
def evaluate_model(model, dataloader, task):
    model.eval()
    total, correct, total_loss = 0, 0, 0
    criterion = nn.MSELoss() if task == 'age' else (nn.BCELoss() if task=='gender' else nn.CrossEntropyLoss())
    with torch.no_grad():
        for imgs, ages, genders, races in dataloader:
            imgs = imgs.to(DEVICE)
            if task == 'age':
                labels = ages.unsqueeze(1).to(DEVICE)
            elif task == 'gender':
                labels = genders.unsqueeze(1).to(DEVICE)
            elif task == 'race':
                labels = races.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            if task == 'gender':
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
            elif task == 'race':
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
            total += imgs.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total if task in ['gender','race'] else None
    return avg_loss, accuracy

# ============================
# Main pipeline
# ============================
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Loading dataset...")
    image_paths, y_age, y_gender, y_race = load_dataset()
    
    # Split dataset
    X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test, y_race_train, y_race_test = train_test_split(
        image_paths, y_age, y_gender, y_race, test_size=0.2, random_state=42)
    X_train, X_val, y_age_train, y_age_val, y_gender_train, y_gender_val, y_race_train, y_race_val = train_test_split(
        X_train, y_age_train, y_gender_train, y_race_train, test_size=0.1, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = UTKFaceDataset(X_train, y_age_train, y_gender_train, y_race_train, transform=transform)
    val_dataset   = UTKFaceDataset(X_val, y_age_val, y_gender_val, y_race_val, transform=transform)
    test_dataset  = UTKFaceDataset(X_test, y_age_test, y_gender_test, y_race_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ====== Age model (regression) ======
    age_model = CNNModel(1, task_type='regression').to(DEVICE)
    age_optimizer = optim.Adam(age_model.parameters(), lr=0.001)
    age_criterion = nn.MSELoss()
    print("Training age model...")
    for epoch in range(EPOCHS):
        loss = train_model(age_model, train_loader, age_criterion, age_optimizer, task='age')
        print(f"Epoch {epoch+1}/{EPOCHS}, Age Loss: {loss:.4f}")
    torch.save(age_model.state_dict(), os.path.join(MODEL_DIR, "age_model.pth"))

    # ====== Gender model (binary) ======
    gender_model = CNNModel(1, task_type='classification').to(DEVICE)
    gender_optimizer = optim.Adam(gender_model.parameters(), lr=0.001)
    gender_criterion = nn.BCELoss()
    print("Training gender model...")
    for epoch in range(EPOCHS):
        loss = train_model(gender_model, train_loader, gender_criterion, gender_optimizer, task='gender')
        print(f"Epoch {epoch+1}/{EPOCHS}, Gender Loss: {loss:.4f}")
    torch.save(gender_model.state_dict(), os.path.join(MODEL_DIR, "gender_model.pth"))

    # ====== Race model (multiclass) ======
    race_model = CNNModel(NUM_RACES, task_type='classification').to(DEVICE)
    race_optimizer = optim.Adam(race_model.parameters(), lr=0.001)
    race_criterion = nn.CrossEntropyLoss()
    print("Training race model...")
    for epoch in range(EPOCHS):
        loss = train_model(race_model, train_loader, race_criterion, race_optimizer, task='race')
        print(f"Epoch {epoch+1}/{EPOCHS}, Race Loss: {loss:.4f}")
    torch.save(race_model.state_dict(), os.path.join(MODEL_DIR, "race_model.pth"))

    # ====== Evaluation ======
    print("Evaluating models...")
    age_loss, _ = evaluate_model(age_model, test_loader, 'age')
    gender_loss, gender_acc = evaluate_model(gender_model, test_loader, 'gender')
    race_loss, race_acc = evaluate_model(race_model, test_loader, 'race')
    print(f"Test Age Loss: {age_loss:.4f}")
    print(f"Test Gender Loss: {gender_loss:.4f}, Accuracy: {gender_acc:.4f}")
    print(f"Test Race Loss: {race_loss:.4f}, Accuracy: {race_acc:.4f}")

if __name__ == "__main__":
    main()
