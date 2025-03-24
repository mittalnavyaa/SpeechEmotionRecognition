import torch
import torch.nn as nn
import timm
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image
from pathlib import Path
from torchvision import transforms

# Constants
NUM_CLASSES = 8  # RAVDESS has 8 emotions
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

class EmotionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class CQTDataset(Dataset):
    def __init__(self, cqt_dir, transform=None):
        self.cqt_dir = Path(cqt_dir)
        self.transform = transform
        self.samples = []
        
        # Collect all file paths and labels
        for actor_dir in sorted(self.cqt_dir.glob("Actor_*")):
            for spec_file in actor_dir.glob("*_cqt.png"):
                emotion = int(spec_file.stem.split("-")[2]) - 1
                self.samples.append((spec_file, emotion))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Load and convert image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        else:
            # Default transform if none provided
            transform = transforms.Compose([
                transforms.Resize((299, 299)),  # Xception input size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
            img = transform(img)
            
        return img, label

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in tqdm(train_loader, desc="Training"):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / len(train_loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return running_loss / len(loader), correct / total, all_preds, all_labels

def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_dir):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_metrics.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_dir):
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotions, yticklabels=emotions)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()

def plot_train_test_accuracy(train_accs, test_accs, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Train Accuracy', marker='o')
    plt.plot(test_accs, label='Test Accuracy', marker='s')
    plt.title('Training vs Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/train_test_accuracy.png')
    plt.close()

def plot_detailed_confusion_matrix(y_true, y_pred, save_dir):
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot with both counts and percentages
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotions, yticklabels=emotions)
    plt.title('Confusion Matrix (Counts)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix_counts.png')
    plt.close()
    
    # Plot percentage confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlBu_r',
                xticklabels=emotions, yticklabels=emotions)
    plt.title('Confusion Matrix (Percentages)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix_percentages.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup transforms for Xception
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Xception input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    print("Loading CQT spectrograms...")
    dataset = CQTDataset(
        r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\RAVDESS_CQT_spectrograms",
        transform=transform
    )
    print(f"Found {len(dataset)} spectrograms")
    
    # Split indices for train/val/test
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)
    
    # Initialize model
    print("Initializing Xception model...")
    model = timm.create_model('xception', pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # Create output directory
    output_dir = Path(r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\RAVDESS_CQT_results")
    output_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    test_accs = []
    
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train and evaluate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Test Acc: {test_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best_xception_cqt_model.pth')
    
    # Plot results
    plot_metrics(train_losses, val_losses, train_accs, val_accs, output_dir)
    plot_train_test_accuracy(train_accs, test_accs, output_dir)
    plot_detailed_confusion_matrix(test_labels, test_preds, output_dir)
    
    # Final evaluation
    model.load_state_dict(torch.load(output_dir / 'best_xception_cqt_model.pth'))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    print("\nFinal Results:")
    print(f"Best Test Accuracy: {max(test_accs):.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    print("\nClassification Report:")
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    print(classification_report(test_labels, test_preds, target_names=emotions))

if __name__ == "__main__":
    main()