import os
import torch
import timm
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import Dataset, DataLoader

class RAVDESSDataset(Dataset):
    def __init__(self, spectrogram_dir, transform=None):
        self.spectrogram_dir = Path(spectrogram_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Load all spectrograms and their labels
        for actor_dir in sorted(self.spectrogram_dir.glob("Actor_*")):
            for spec_file in actor_dir.glob("*_melspec.png"):
                self.samples.append(spec_file)
                # Extract emotion from filename (03-01-01-01-01-01-01_melspec.png)
                emotion = int(spec_file.stem.split("-")[2]) - 1
                self.labels.append(emotion)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load and convert image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        return img, label

def preprocess_ravdess():
    # Setup paths
    spectrogram_dir = Path(r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\RAVDESS_spectrograms")
    output_dir = Path(r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\RAVDESS_processed")
    output_dir.mkdir(exist_ok=True)

    # Load Xception model and get preprocessing transform
    print("Loading Xception model...")
    model = timm.create_model('xception', pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    # Create dataset and dataloader
    dataset = RAVDESSDataset(spectrogram_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"Found {len(dataset)} spectrograms")

    # Process all spectrograms
    processed_features = []
    all_labels = []
    
    print("Processing spectrograms...")
    for batch_imgs, batch_labels in tqdm(dataloader):
        processed_features.append(batch_imgs)
        all_labels.append(batch_labels)

    # Concatenate all batches
    processed_features = torch.cat(processed_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Save processed data
    print("Saving processed data...")
    torch.save({
        'features': processed_features,
        'labels': all_labels,
        'filenames': [str(path) for path in dataset.samples]
    }, output_dir / 'ravdess_xception_features.pt')

    # Print statistics
    print("\nProcessing complete:")
    print(f"Total samples processed: {len(dataset)}")
    print(f"Feature tensor shape: {processed_features.shape}")
    print("\nEmotion distribution:")
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    for emotion_idx, emotion_name in enumerate(emotions):
        count = (all_labels == emotion_idx).sum().item()
        print(f"  {emotion_name}: {count} samples")

if __name__ == "__main__":
    preprocess_ravdess()