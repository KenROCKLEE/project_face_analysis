"""
UTKFace Age Classification with Improved ResNet-50
Author: Your Name
Date: 2025-08-23
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# ----------------- CONFIG -----------------
IMG_SIZE = (224, 224)  # Increased for ResNet-50
AGE_DIR = 'datasets/clean/age'
MODEL_DIR = 'models'
BATCH_SIZE = 32
EPOCHS_PHASE1 = 15  # Train only the head
EPOCHS_PHASE2 = 50  # Fine-tune the full model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATIENCE = 20  # Increased patience

# Age groups
AGE_GROUPS = [
    (0, 6),
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

# Enhanced Transforms with more aggressive augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    normalize_transform
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
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

def split_dataset(X, y, test_size=0.15, val_size=0.15, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                       random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, 
                                                     random_state=random_state, stratify=y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_weighted_sampler(y_train):
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler

# ----------------- MODEL -----------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class ResNetAgeClassification(nn.Module):
    def __init__(self, num_classes=NUM_AGE_CLASSES):
        super().__init__()
        # Use ResNet-50 for more capacity
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        num_features = self.resnet.fc.in_features
        # Enhanced classifier head
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# ----------------- TRAINING -----------------
def train_age_model(model, train_loader, val_loader, y_train, model_name='age_model_resnet.pth'):
    model.to(DEVICE)
    
    # Label smoothing instead of class weights
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    best_val_loss = float('inf')
    best_val_acc = 0
    best_model_state = None
    train_losses, val_losses, val_accuracies = [], [], []
    
    # --- PHASE 1: Train the classifier head only ---
    print("\n--- Starting Phase 1: Training classifier head ---")
    for param in model.resnet.parameters():
        param.requires_grad = False
    for param in model.resnet.fc.parameters():
        param.requires_grad = True

    optimizer_phase1 = optim.AdamW(model.resnet.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase1, EPOCHS_PHASE1)

    for epoch in range(1, EPOCHS_PHASE1 + 1):
        model.train()
        running_loss = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            with torch.cuda.amp.autocast():
                output = model(xb)
                loss = criterion(output, yb)
            
            optimizer_phase1.zero_grad()
            loss.backward()
            optimizer_phase1.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss, val_acc = validate_model(model, val_loader, criterion)
        
        scheduler_phase1.step()
        
        print(f"Phase 1 - Epoch {epoch}/{EPOCHS_PHASE1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer_phase1.param_groups[0]['lr']:.2e}")

    # --- PHASE 2: Fine-tune the full model ---
    print("\n--- Starting Phase 2: Fine-tuning ResNet layers ---")
    # Unfreeze later layers gradually
    for param in model.resnet.layer3.parameters():
        param.requires_grad = True
    for param in model.resnet.layer4.parameters():
        param.requires_grad = True

    # Layer-specific learning rates
    optimizer_phase2 = optim.AdamW([
        {'params': model.resnet.layer3.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},
        {'params': model.resnet.layer4.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},
        {'params': model.resnet.fc.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4}
    ])
    
    # Learning rate scheduling with warmup
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer_phase2, start_factor=0.01, total_iters=5)
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=EPOCHS_PHASE2-5)
    
    scaler = torch.cuda.amp.GradScaler()
    patience_counter = 0

    for epoch in range(1, EPOCHS_PHASE2 + 1):
        model.train()
        running_loss = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            with torch.cuda.amp.autocast():
                output = model(xb)
                loss = criterion(output, yb)
            
            optimizer_phase2.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer_phase2)
            scaler.update()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss, val_acc = validate_model(model, val_loader, criterion)
        
        # Learning rate scheduling
        if epoch <= 5:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        current_lr = optimizer_phase2.param_groups[0]['lr']
        print(f"Phase 2 - Epoch {epoch}/{EPOCHS_PHASE2} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}")

        # Early stopping with patience
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"↗ New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"🛑 Early stopping at epoch {epoch}")
            break

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

    # Save best model
    os.makedirs(MODEL_DIR, exist_ok=True)
    if best_model_state:
        torch.save(best_model_state, os.path.join(MODEL_DIR, model_name))
        print(f"\n💾 Best model saved with Val Acc: {best_val_acc:.4f}, Val Loss: {best_val_loss:.4f}")

    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Val Loss', alpha=0.8)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green', alpha=0.8)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def validate_model(model, loader, criterion):
    model.eval()
    val_loss = 0
    val_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            with torch.cuda.amp.autocast():
                output = model(xb)
                loss = criterion(output, yb)
            
            pred = output.argmax(dim=1)
            val_correct += (pred == yb).sum().item()
            val_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)
    
    avg_val_loss = val_loss / total_samples
    val_acc = val_correct / total_samples
    return avg_val_loss, val_acc

# ----------------- EVALUATION -----------------
def evaluate_age_model(model, test_loader):
    model.to(DEVICE)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            with torch.cuda.amp.autocast():
                output = model(xb)
            
            pred = output.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    
    acc = correct / total
    print(f"\n🎯 Test Accuracy: {acc:.4f} ({correct}/{total})")
    return acc

def test_time_augmentation(model, test_loader, n_augments=5):
    """Test Time Augmentation for improved evaluation"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            # Original image prediction
            output_original = model(xb)
            
            # Augmented predictions
            augmented_outputs = []
            for _ in range(n_augments):
                augmented_batch = torch.stack([train_transforms(transforms.ToPILImage()(img.cpu())) 
                                             for img in xb]).to(DEVICE)
                output_aug = model(augmented_batch)
                augmented_outputs.append(output_aug)
            
            # Average predictions
            all_outputs = torch.stack([output_original] + augmented_outputs)
            final_output = all_outputs.mean(dim=0)
            
            pred = final_output.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    
    tta_acc = correct / total
    print(f"🎯 TTA Test Accuracy ({n_augments} augments): {tta_acc:.4f} ({correct}/{total})")
    return tta_acc

# ----------------- MAIN -----------------
def main():
    if torch.cuda.is_available():
        print(f"✅ GPU is available and will be used! Device: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA version: {torch.version.cuda}")
    else:
        print("⚠️ No GPU found, using CPU")

    # Load data
    X_age, y_age = load_age_dataset()
    print(f"📊 Total samples: {len(X_age)}")
    print(f"📊 Class distribution: {np.bincount(y_age)}")
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X_age, y_age)
    print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create weighted sampler for training
    sampler = create_weighted_sampler(y_train)
    
    # Create dataloaders
    train_dataset = AgeDatasetClassification(X_train, y_train, transform=train_transforms)
    val_dataset = AgeDatasetClassification(X_val, y_val, transform=val_transforms)
    test_dataset = AgeDatasetClassification(X_test, y_test, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = ResNetAgeClassification()
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_age_model(model, train_loader, val_loader, y_train, 'age_model_resnet_best.pth')
    
    # Load best model and evaluate
    best_model_path = os.path.join(MODEL_DIR, 'age_model_resnet_best.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        print("✅ Loaded best model for final evaluation.")
        
        # Standard evaluation
        test_acc = evaluate_age_model(model, test_loader)
        
        # Test Time Augmentation evaluation
        tta_acc = test_time_augmentation(model, test_loader, n_augments=5)
        
        print(f"\n📈 Performance Summary:")
        print(f"   • Standard Test Accuracy: {test_acc:.4f}")
        print(f"   • TTA Test Accuracy: {tta_acc:.4f}")
        print(f"   • Improvement with TTA: {tta_acc - test_acc:.4f}")
        
    else:
        print("❌ Best model not found. Cannot perform final evaluation.")

    print("\n🎉 Age classification model training completed!")

if __name__ == '__main__':
    main()