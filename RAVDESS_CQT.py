import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
from pathlib import Path

class RAVDESSPreprocessor:
    def __init__(self, sample_rate=16000, hop_length=512, n_bins=84, bins_per_octave=12):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave

    def create_cqt_spectrogram(self, audio_path, output_path):
        # Load audio using librosa
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Generate CQT
        cqt = librosa.cqt(
            waveform, 
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave
        )
        
        # Convert to dB scale
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        
        # Normalize
        cqt_norm = ((cqt_db - cqt_db.min()) / 
                    (cqt_db.max() - cqt_db.min()) * 255).astype(np.uint8)
        
        # Create and save plot
        plt.figure(figsize=(10, 4))
        plt.imshow(cqt_norm, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'CQT Spectrogram - {Path(audio_path).stem}')
        plt.ylabel('Frequency Bin')
        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return cqt_norm

def process_ravdess():
    # Setup paths
    input_path = Path(r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\RAVDESS")
    output_base = Path(r"C:\Users\navya\wav2vec2-lg-xlsr-en-speech-emotion-recognition\RAVDESS_CQT_spectrograms")
    
    # Create output directories
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create preprocessor with CQT parameters
    preprocessor = RAVDESSPreprocessor(
        sample_rate=16000,
        hop_length=512,
        n_bins=84,
        bins_per_octave=12
    )
    
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
                output_path = actor_output_dir / f"{wav_path.stem}_cqt.png"
                
                # Generate and save CQT spectrogram
                preprocessor.create_cqt_spectrogram(str(wav_path), str(output_path))
                
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
        f.write(f"RAVDESS CQT Processing Report\n")
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