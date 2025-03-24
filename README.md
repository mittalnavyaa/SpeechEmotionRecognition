# File Details

## 1. preprocess_RAVDESS.py
This script converts the [RAVDESS dataset](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) into a format compatible with the speech emotion recognition model in the same directory. It processes and normalizes audio files, converts labels, and outputs preprocessed data for direct use with the Wav2Vec2 model.

## 2. embeddings_RAVDESS.npy
Contains embeddings for the Wav2Vec2 model.

## 3. labels_RAVDESS.npy
Contains labels of the Wav2Vec2 model.

## 4. test_RAVDESS.py
Tests the RAVDESS data on the fully connected feed-forward neural network pipeline.
