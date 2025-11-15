# File: src/01_preprocess_data.py

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
AUDIO_DIR = 'data/'
OUTPUT_DIR = 'processed_data/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Spectrogram parameters
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

# Chunking parameters
CHUNK_SECONDS = 4
SAMPLES_PER_CHUNK = SAMPLE_RATE * CHUNK_SECONDS

def parse_filename(filename):
    """Extracts the 4-rotor state from the filename."""
    parts = filename.split('.')[0].split('_')
    if len(parts) == 2 and len(parts[1]) == 4:
        return list(parts[1])
    return None

def process_data():
    """Loads audio, creates spectrogram chunks, and saves them to disk."""
    print("Starting data preprocessing...")
    
    # --- Step 1: Create a Data Manifest ---
    file_data = []
    for filename in os.listdir(AUDIO_DIR):
        if filename.endswith('.wav'):
            labels = parse_filename(filename)
            if labels:
                file_path = os.path.join(AUDIO_DIR, filename)
                file_data.append({'path': file_path, 'labels': labels})
    
    manifest_df = pd.DataFrame(file_data)
    label_map = {'H': 0, 'F': 1, 'E': 2}
    manifest_df['encoded_labels'] = manifest_df['labels'].apply(
        lambda labels: [label_map[l] for l in labels]
    )
    print(f"Found {len(manifest_df)} audio files to process.")

    # --- Step 2: The Core Processing Loop ---
    X = []
    y = []

    for index, row in tqdm(manifest_df.iterrows(), total=manifest_df.shape[0], desc="Processing files"):
        file_path = row['path']
        encoded_labels = row['encoded_labels']

        try:
            y_audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=False)
            
            for i in range(0, y_audio.shape[1] - SAMPLES_PER_CHUNK + 1, SAMPLES_PER_CHUNK):
                chunk = y_audio[:, i : i + SAMPLES_PER_CHUNK]
                
                all_channels_mel = []
                for channel in range(chunk.shape[0]):
                    mel_spec = librosa.feature.melspectrogram(
                        y=chunk[channel], sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
                    )
                    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                    all_channels_mel.append(log_mel_spec)

                multi_channel_spec = np.stack(all_channels_mel, axis=-1)
                X.append(multi_channel_spec)
                y.append(encoded_labels)

        except Exception as e:
            print(f"\nError processing {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"\nFinished processing.")
    print(f"Shape of our data (X): {X.shape}")
    print(f"Shape of our labels (y): {y.shape}")

    # --- Step 3: Save Processed Data ---
    np.save(os.path.join(OUTPUT_DIR, 'X_data.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y_labels.npy'), y)
    print(f"Data saved to {OUTPUT_DIR}")

    return X

def visualize_sample(X_data):
    """Plots a sanity-check visualization of one sample."""
    print("\nVisualizing one sample for a sanity check...")
    sample_data = X_data[0]
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle('Spectrograms for a Single 4-Second Chunk')
    
    for i in range(4):
        ax = axs[i // 2, i % 2]
        img = librosa.display.specshow(sample_data[:, :, i], sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title(f'Rotor {i+1} (Mic Channel {i})')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    processed_X = process_data()
    visualize_sample(processed_X)