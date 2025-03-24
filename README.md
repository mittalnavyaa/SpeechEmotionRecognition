# File Details

## 1. preprocess_RAVDESS.py
This script converts the [RAVDESS dataset](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) into a format compatible with the speech emotion recognition model in the same directory. It processes and normalizes audio files, converts labels, and outputs preprocessed data for direct use with the Wav2Vec2 model.

## 2. embeddings_RAVDESS.npy
Contains embeddings for the Wav2Vec2 model.

## 3. labels_RAVDESS.npy
Contains labels of the Wav2Vec2 model.

## 4. test_RAVDESS.py
Tests the RAVDESS data on the fully connected feed-forward neural network pipeline.

## 5. preprocess_CREMAD.py
Processes the CREMA-D dataset to prepare it for input into the speech emotion recognition model by extracting features and normalizing audio data.

## 6. embeddings_CREMAD.npy
Contains extracted embeddings for the CREMA-D dataset using the Wav2Vec2 model.

## 7. labels_CREMAD.npy
Contains emotion labels corresponding to the CREMA-D dataset.

## 8. test_CREMAD.py
Tests the CREMA-D dataset on the fully connected feed-forward neural network pipeline.

## 9. preprocess_MELD.py
Processes the MELD dataset by extracting and normalizing features to prepare the data for speech emotion recognition.

## 10. embeddings_train_MELD.npy
Contains extracted embeddings for the training set of the MELD dataset.

## 11. embeddings_dev_MELD.npy
Contains extracted embeddings for the development set of the MELD dataset.

## 12. embeddings_test_MELD.npy
Contains extracted embeddings for the test set of the MELD dataset.

## 13. labels_train_MELD.npy
Contains emotion labels for the training set of the MELD dataset.

## 14. labels_dev_MELD.npy
Contains emotion labels for the development set of the MELD dataset.

## 15. labels_test_MELD.npy
Contains emotion labels for the test set of the MELD dataset.

## 16. test_MELD.py
Tests the MELD dataset on the fully connected feed-forward neural network pipeline.

## 17. RAVDESS_mel.py
Converts the RAVDESS dataset into Mel spectrograms.

## 18. preprocess_RAVDESS_Xception.py
Preprocesses the RAVDESS Mel spectrogram dataset for the Xception model.

## 19. Xception_RAVDESS_testing.py
Tests the preprocessed RAVDESS Mel spectrogram dataset.

## 20. RAVDESS_CQT.py
Converts the RAVDESS dataset into CQT (Constant-Q Transform) form.

## 21. Xception_RAVDESS_CQT_testing.py
Tests the CQT dataset on the Xception model.
