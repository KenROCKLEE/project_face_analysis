"""
Emotion Prediction Pipeline in PyTorch
Author: Your Name
Date: 2025-08-22

This script trains a CNN model for emotion detection using the cleaned FER-2013 dataset.
Classes: angry, happy, neutral, sad, surprise
- Balances every class by undersampling to the size of the smallest class.
- Input size: 128x128
- Batch size: 32
- Saves the best model based on validation accuracy
- Incorporates ResNet-18 for transfer learning and refined fine-tuning
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
from collections import Counter
import random 

# ----------------- CONFIG -----------------
IMG_SIZE = (128, 128)
EMOTION_DIR = 'datasets/clean/emotion'
MODEL_DIR = 'models'
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Emotion classes and mapping (dynamically determined)
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
    filepaths_by_class = {emotion: [] for emotion in EMOTION_CLASSES}
    
    print("Loading emotion data from cleaned directory...")
    
    # First, collect all filepaths per class
    for folder_name in EMOTION_CLASSES:
        class_dir = os.path.join(emotion_dir, folder_name)
        for fname in os.listdir(class_dir):
            if fname.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(class_dir, fname)
                filepaths_by_class[folder_name].append(img_path)

    # Determine the minimum number of samples in any class
    min_samples = float('inf')
    for emotion_class in EMOTION_CLASSES:
        min_samples = min(min_samples, len(filepaths_by_class[emotion_class]))
    
    print(f"Balancing dataset to {min_samples} samples per class.")

    X_emotion, y_emotion = [], []
    # Now, sample 'min_samples' from each class
    for folder_name in EMOTION_CLASSES:
        random.shuffle(filepaths_by_class[folder_name]) # Shuffle to get random samples
        selected_filepaths = filepaths_by_class[folder_name][:min_samples]
        emotion_label = EMOTION_MAP[folder_name]
        
        for img_path in selected_filepaths:
            img = Image.open(img_path).convert('RGB').resize(img_size)
            X_emotion.append(img)
            y_emotion.append(emotion_label)

    # Return as lists, not numpy arrays, to preserve PIL Image type
    return X_emotion, y_emotion


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

def create_dataloader(X, y, batch_size=BATCH_SIZE, transforms=None, shuffle=True):
    dataset = EmotionDataset(X, y, transform=transforms)
    # When all classes are already balanced via undersampling, a WeightedRandomSampler is not strictly needed,
    # and simple shuffling is sufficient.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
        
        # This attribute is not directly used in the current train_model, but kept for context.
        self.freeze_until = "fc" 

    def forward(self, x):
        return self.resnet(x)

# ----------------- TRAIN -----------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=1e-3, phase_name="Training", model_name='emotion_model.pth'):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_accuracy = 0
    patience_counter = 0
    best_model_state = None

    train_losses, val_losses, val_accuracies = [], [], []

    # Ensure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    full_model_path = os.path.join(MODEL_DIR, model_name)

    print(f"\n--- {phase_name} ---")
    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        running_loss = 0.0
        # correct_train = 0 # Not used for train loss calculation, removed for clarity

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_val_samples += yb.size(0)
                correct_val += (predicted == yb).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = correct_val / total_val_samples # Corrected total samples for accuracy
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {avg_val_accuracy:.4f}")
        
        scheduler.step(avg_val_loss)

        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict()
            print(f"Model saved! Val Accuracy improved to {best_val_accuracy:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= 10: # Early stopping patience
            print(f"Early stopping at epoch {epoch} due to no improvement in Val Accuracy.")
            break
            
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss) # Corrected variable name from avg_loss to avg_val_loss
        val_accuracies.append(avg_val_accuracy)

    if best_model_state:
        torch.save(best_model_state, full_model_path)
        print(f"✅ Training complete. Best model saved to {full_model_path}")
    else: # Fallback if no improvement (e.g., very short training, or bad params)
        torch.save(model.state_dict(), full_model_path)
        print(f"Training complete. No improvement found, last model state saved to {full_model_path}")

    return train_losses, val_losses, val_accuracies

# ----------------- EVALUATION -----------------
def evaluate_model(model, test_loader):
    model.to(DEVICE)
    model.eval()
    correct = 0
    total_test_samples = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb)
            _, preds = torch.max(output, 1)
            total_test_samples += yb.size(0)
            correct += (preds == yb).sum().item()
    
    test_accuracy = correct / total_test_samples
    print(f"Test Accuracy: {test_accuracy:.4f}")

# ----------------- MAIN -----------------
def main():
    if torch.cuda.is_available():
        print(f"✅ GPU is available and will be used! Device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU found, using CPU")

    print("Loading emotion dataset...")
    X_emotion, y_emotion = load_emotion_dataset()
    
    # Print class distribution after balancing
    unique_classes, counts = np.unique(y_emotion, return_counts=True)
    print("Balanced Class Distribution:", dict(zip(unique_classes, counts)))
    print(f"Total balanced emotion samples: {len(X_emotion)}")
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X_emotion, y_emotion)
    
    # Dataloaders - No WeightedRandomSampler needed as data is already balanced
    train_loader = create_dataloader(X_train, y_train, transforms=train_transforms, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, transforms=val_transforms, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, transforms=val_transforms, shuffle=False)
    
    # Train Emotion model
    print("\nTraining Emotion model with ResNet...")
    emotion_model = ResNetEmotionModel(output_units=NUM_EMOTION_CLASSES)
    
    # Phase 1: Feature Extraction
    print("\nPhase 1: Training only the top layers (Feature Extraction)")
    phase1_train_losses, phase1_val_losses, phase1_val_accuracies = train_model(
        emotion_model, train_loader, val_loader, epochs=10, lr=1e-3, 
        phase_name="Phase 1: Feature Extraction", model_name='emotion_model_resnet_phase1.pth' # Save interim model
    )
    # Plot graphs for Phase 1
    plot_history(phase1_train_losses, phase1_val_losses, phase1_val_accuracies, 
                 "Phase 1: Feature Extraction", MODEL_DIR)

    # Phase 2: Fine-tuning
    print("\nPhase 2: Fine-tuning the entire model")
    # Load the best model state from Phase 1 to continue training from the best point
    emotion_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'emotion_model_resnet_phase1.pth')))

    # Unfreeze specific layers for fine-tuning: layer3, layer4 and the fc layer
    for name, param in emotion_model.resnet.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False # Ensure earlier layers remain frozen
    
    phase2_train_losses, phase2_val_losses, phase2_val_accuracies = train_model(
        emotion_model, train_loader, val_loader, epochs=EPOCHS, lr=1e-5, # Use EPOCHS and lower LR
        phase_name="Phase 2: Fine-tuning", model_name='emotion_model_resnet.pth' # Final model name
    )
    # Plot graphs for Phase 2
    plot_history(phase2_train_losses, phase2_val_losses, phase2_val_accuracies, 
                 "Phase 2: Fine-tuning", MODEL_DIR)

    # Evaluate final model
    print("\nEvaluating final model on test set...")
    best_model = ResNetEmotionModel(output_units=NUM_EMOTION_CLASSES)
    best_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'emotion_model_resnet.pth')))
    evaluate_model(best_model, test_loader)
    
    print("All models trained, evaluated, and saved successfully!")

# =========================
# Plotting Function (for individual phases)
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

if __name__ == '__main__':
    main()
