import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF 
import matplotlib.pyplot as plt
import numpy as np
import copy 

# --- 1. Basic Classes (Keep unchanged) ---
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
             tensor = transforms.ToTensor()(tensor)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class BreastDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img = self.images[idx]
        img = transforms.ToPILImage()(img)
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128), 
            nn.ReLU(),
            nn.Dropout(0.6), 
            nn.Linear(128, 2)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 2. Core Improvement Logic (Fixed verbose error) ---
def run_cnn(x_train, y_train, x_val, y_val, x_test, y_test, device, save_dir):
    print("\n" + "="*50)
    print("[MODEL B PRO] Training with Advanced Techniques...")
    print("Strategies: LR Scheduler, Weight Decay, ColorJitter, Best Model Saving")
    print("="*50)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # === Strategy 1: Enhanced Data Augmentation ===
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5), 
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.05),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_loader = DataLoader(BreastDataset(x_train, y_train, train_transform), batch_size=32, shuffle=True)
    val_loader = DataLoader(BreastDataset(x_val, y_val, test_transform), batch_size=32, shuffle=False)
    test_loader = DataLoader(BreastDataset(x_test, y_test, test_transform), batch_size=32, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # === Strategy 2: Weight Decay (L2 Regularization) ===
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # === Strategy 3: Learning Rate Scheduler ===
    # Fix: Removed verbose=True parameter to avoid compatibility issues
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    train_losses, val_accs = [], []
    epochs = 25 
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict()) 

    print(f"Start Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0 
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        # Update Learning Rate
        scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss)
        val_accs.append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        # Manually print LR info, replacing verbose=True functionality
        print(f"Epoch {epoch+1:02d}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val Acc={val_acc:.4f} | LR={current_lr:.1e}")

        # === Strategy 4: Save Best Model ===
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"  >>> New Best Accuracy! Model Saved.")

    print(f"\nTraining Finished. Best Validation Accuracy: {best_acc:.4f}")
    
    print("Loading best model weights for final testing...")
    model.load_state_dict(best_model_wts)

    # Plot Curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title(f"Training Dynamics (Best Val Acc: {best_acc:.4f})")
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'Model_B_Training_Curves_Pro.png'))
    plt.close()

    # Final Testing
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images), 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\n[Model B Pro Result] Final Test Set Evaluation")
    from sklearn.metrics import classification_report, accuracy_score
    final_acc = accuracy_score(all_labels, all_preds)
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))
    
    return final_acc