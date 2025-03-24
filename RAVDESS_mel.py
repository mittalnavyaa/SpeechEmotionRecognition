import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

class RAVDESSPreprocessor:
    def __init__(self, sample_rate=16000, n_mels=128):
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def create_melspectrogram(self, audio_path, output_path):
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Generate mel spectrogram
        mel_spect = self.mel_transform(waveform)
        mel_spect_db = self.amplitude_to_db(mel_spect)
        
        # Convert to numpy and normalize
        mel_spect_np = mel_spect_db.squeeze().numpy()
        mel_spect_np = ((mel_spect_np - mel_spect_np.min()) / 
                       (mel_spect_np.max() - mel_spect_np.min()) * 255).astype(np.uint8)
        
        # Create and save plot
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spect_np, aspect='auto', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram - {Path(audio_path).stem}')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return mel_spect_np

def process_ravdess():
    # Setup paths
    input_path = Path(r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\RAVDESS")
    output_base = Path(r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\RAVDESS_spectrograms")
    
    # Create output directories
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create preprocessor
    preprocessor = RAVDESSPreprocessor()
    
    # Initialize counters
    total_processed = 0
    failed_files = []
    emotion_counts = {i: 0 for i in range(8)}
    
    # Process all actors
    for actor_id in range(1, 25):  # RAVDESS has 24 actors
        actor_dir = input_path / f"Actor_{actor_id:02d}"
        if not actor_dir.exists():
            print(f"Warning: Directory for Actor_{actor_id:02d} not found")
            continue
        
        # Create actor output directory
        actor_output_dir = output_base / f"Actor_{actor_id:02d}"
        actor_output_dir.mkdir(exist_ok=True)
        
        # Get all wav files for current actor
        wav_files = list(actor_dir.glob("*.wav"))
        if not wav_files:
            print(f"Warning: No WAV files found for Actor_{actor_id:02d}")
            continue
            
        print(f"\nProcessing Actor_{actor_id:02d} ({len(wav_files)} files)")
        
        # Process each wav file
        for wav_path in tqdm(wav_files, desc=f"Actor {actor_id:02d}"):
            try:
                # Extract emotion from filename (format: 03-01-01-01-01-01-01.wav)
                emotion = int(wav_path.stem.split("-")[2]) - 1
                
                # Create output path
                output_path = actor_output_dir / f"{wav_path.stem}_melspec.png"
                
                # Generate and save spectrogram
                preprocessor.create_melspectrogram(str(wav_path), str(output_path))
                
                total_processed += 1
                emotion_counts[emotion] += 1
                
            except Exception as e:
                failed_files.append((str(wav_path), str(e)))
                print(f"Error processing {wav_path.name}: {str(e)}")
    
    # Print final statistics
    print("\nProcessing Complete:")
    print(f"Total files processed successfully: {total_processed}")
    print(f"Failed files: {len(failed_files)}")
    
    print("\nEmotion distribution:")
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    for emotion, count in emotion_counts.items():
        print(f"Emotion {emotion + 1} ({emotions[emotion]}): {count} files")
    
    if failed_files:
        print("\nFailed files details:")
        for file, error in failed_files:
            print(f"- {file}: {error}")
            
    # Save processing report
    with open(output_base / "processing_report.txt", "w") as f:
        f.write(f"RAVDESS Processing Report\n")
        f.write(f"Total files processed: {total_processed}\n")
        f.write(f"Failed files: {len(failed_files)}\n\n")
        f.write("Emotion distribution:\n")
        for emotion, count in emotion_counts.items():
            f.write(f"Emotion {emotion + 1} ({emotions[emotion]}): {count} files\n")
        if failed_files:
            f.write("\nFailed files:\n")
            for file, error in failed_files:
                f.write(f"- {file}: {error}\n")

if __name__ == "__main__":
    process_ravdess()