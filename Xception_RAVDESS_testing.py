import torch
import torch.nn as nn
import timm
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns

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
    
    # Load preprocessed data
    data = torch.load('RAVDESS_processed/ravdess_xception_features.pt')
    features = data['features']
    labels = data['labels']
    
    # Split data
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features, train_labels, test_size=0.25, random_state=42, stratify=train_labels
    )
    
    # Create datasets and dataloaders
    train_dataset = EmotionDataset(train_features, train_labels)
    val_dataset = EmotionDataset(val_features, val_labels)
    test_dataset = EmotionDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = timm.create_model('xception', pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # Training loop
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    test_accs = []  # Add list for test accuracies
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        
        # Test during training
        test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
        test_accs.append(test_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Test Acc: {test_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'RAVDESS_processed/best_xception_model.pth')
    
    # Plot results
    plot_metrics(train_losses, val_losses, train_accs, val_accs, 'RAVDESS_processed')
    plot_train_test_accuracy(train_accs, test_accs, 'RAVDESS_processed')
    plot_detailed_confusion_matrix(test_labels, test_preds, 'RAVDESS_processed')
    
    # Final evaluation and metrics
    model.load_state_dict(torch.load('RAVDESS_processed/best_xception_model.pth'))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    print("\nFinal Results:")
    print(f"Best Test Accuracy: {max(test_accs):.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Print classification report
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=emotions))

if __name__ == "__main__":
    main()