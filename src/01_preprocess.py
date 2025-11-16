import os
import numpy as np                                          # type: ignore
import pandas as pd                                         # type: ignore
import librosa                                              # type: ignore
import librosa.display                                      # type: ignore
import matplotlib.pyplot as plt                             # type: ignore
from tqdm import tqdm                                       # type: ignore
from sklearn.model_selection import train_test_split        # type: ignore

# --- Configuration ---
AUDIO_DIR = 'data/'
OUTPUT_DIR = 'processed_data/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Spectrogram & Chunking parameters
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
CHUNK_SECONDS = 4
SAMPLES_PER_CHUNK = SAMPLE_RATE * CHUNK_SECONDS

def parse_filename(filename):
    """Extracts the 4-rotor state from the filename."""
    parts = filename.split('.')[0].split('_')
    if len(parts) == 2 and len(parts[1]) == 4:
        return list(parts[1])
    return None

def process_data():
    """Loads audio, creates original and augmented spectrogram chunks, and saves them to disk."""
    print("Starting data preprocessing with augmentation...")
    
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

    file_paths = manifest_df['path'].unique()
    train_files = set(train_test_split(file_paths, test_size=0.3, random_state=42)[0])

    print(f"Augmenting data from {len(train_files)} training files.")
    print(f"Processing {len(file_paths) - len(train_files)} test/validation files without augmentation.")

    X = []
    y = []

    # --- Helper function to avoid repeating code ---
    def chunk_to_spec(chunk):
        all_channels_mel = []
        for channel in range(chunk.shape[0]):
            mel_spec = librosa.feature.melspectrogram(
                y=chunk[channel], sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
            )
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            all_channels_mel.append(log_mel_spec)
        return np.stack(all_channels_mel, axis=-1)

    for index, row in tqdm(manifest_df.iterrows(), total=manifest_df.shape[0], desc="Processing files"):
        file_path = row['path']
        encoded_labels = row['encoded_labels']
        is_training_file = file_path in train_files

        try:
            y_audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=False)
            
            for i in range(0, y_audio.shape[1] - SAMPLES_PER_CHUNK + 1, SAMPLES_PER_CHUNK):
                original_chunk = y_audio[:, i : i + SAMPLES_PER_CHUNK]
                
                # Process and add the ORIGINAL chunk
                X.append(chunk_to_spec(original_chunk))
                y.append(encoded_labels)

                # If it's a training file, add AUGMENTED versions
                if is_training_file:
                    # --- AUGMENTATION 1: NOISE (In-line) ---
                    # The fix is here: generate noise with the same shape as the chunk
                    noise = np.random.randn(*original_chunk.shape) * 0.005 
                    chunk_noisy = original_chunk + noise
                    chunk_noisy = chunk_noisy.astype(type(original_chunk[0,0]))
                    X.append(chunk_to_spec(chunk_noisy))
                    y.append(encoded_labels)

                    # --- AUGMENTATION 2: PITCH SHIFT (In-line) ---
                    chunk_pitch_shifted = librosa.effects.pitch_shift(y=original_chunk, sr=sr, n_steps=0.5)
                    # Ensure length is consistent
                    if chunk_pitch_shifted.shape[1] != original_chunk.shape[1]:
                        pad_width = original_chunk.shape[1] - chunk_pitch_shifted.shape[1]
                        chunk_pitch_shifted = np.pad(chunk_pitch_shifted, pad_width=((0,0), (0, pad_width)), mode='constant')
                    
                    X.append(chunk_to_spec(chunk_pitch_shifted))
                    y.append(encoded_labels)
        
        except Exception as e:
            print(f"\nError processing {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"\nFinished processing.")
    print(f"Shape of our new augmented data (X): {X.shape}")
    print(f"Shape of our new labels (y): {y.shape}")

    np.save(os.path.join(OUTPUT_DIR, 'X_data.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y_labels.npy'), y)
    print(f"Data saved to {OUTPUT_DIR}")

    return X

def visualize_sample(X_data):
    """Plots a sanity-check visualization of one sample."""
    if len(X_data) == 0:
        print("No data to visualize.")
        return
        
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