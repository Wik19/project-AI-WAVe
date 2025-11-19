import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- Configuration ---
AUDIO_DIR = 'data/'

# --- EXPERIMENT CONFIGURATION: CHOOSE WHICH MICS TO USE ---
# This is the line you will change for each experimental run.
# Examples:
# [0, 1, 2, 3] -> All four microphones (baseline)
# [0]          -> Only the first microphone
# [0, 2]       -> Microphones 1 and 3
MIC_CHANNELS_TO_USE = [0, 2]

# Automatically create a unique output directory based on the channels used
output_folder_name = "mics_" + "".join(map(str, MIC_CHANNELS_TO_USE))
OUTPUT_DIR = f'processed_data_{output_folder_name}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"--- Starting Experiment: Using Microphones {MIC_CHANNELS_TO_USE} ---")
print(f"Output will be saved to: {OUTPUT_DIR}")

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
    metadata_filenames = []

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
            y_audio_all_channels, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=False)
            
            # --- THIS IS THE KEY MODIFICATION FOR THE ABLATION STUDY ---
            # Select only the microphone channels specified in the config
            y_audio = y_audio_all_channels[MIC_CHANNELS_TO_USE, :]
            
            for i in range(0, y_audio.shape[1] - SAMPLES_PER_CHUNK + 1, SAMPLES_PER_CHUNK):
                original_chunk = y_audio[:, i : i + SAMPLES_PER_CHUNK]
                
                # Process and add the ORIGINAL chunk
                X.append(chunk_to_spec(original_chunk))
                y.append(encoded_labels)
                metadata_filenames.append(file_path)

                # If it's a training file, add AUGMENTED versions
                if is_training_file:
                    noise = np.random.randn(*original_chunk.shape) * 0.005 
                    chunk_noisy = original_chunk + noise
                    chunk_noisy = chunk_noisy.astype(type(original_chunk[0,0]))
                    X.append(chunk_to_spec(chunk_noisy))
                    y.append(encoded_labels)
                    metadata_filenames.append(file_path)

                    chunk_pitch_shifted = librosa.effects.pitch_shift(y=original_chunk, sr=sr, n_steps=0.5)
                    if chunk_pitch_shifted.shape[1] != original_chunk.shape[1]:
                        pad_width = original_chunk.shape[1] - chunk_pitch_shifted.shape[1]
                        chunk_pitch_shifted = np.pad(chunk_pitch_shifted, pad_width=((0,0), (0, pad_width)), mode='constant')
                    X.append(chunk_to_spec(chunk_pitch_shifted))
                    y.append(encoded_labels)
                    metadata_filenames.append(file_path)
        
        except Exception as e:
            print(f"\nError processing {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)
    metadata_filenames = np.array(metadata_filenames)

    print(f"\nFinished processing.")
    print(f"Shape of our new augmented data (X): {X.shape}")
    print(f"Shape of our new labels (y): {y.shape}")
    print(f"Shape of our new metadata: {metadata_filenames.shape}")

    np.save(os.path.join(OUTPUT_DIR, 'X_data.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y_labels.npy'), y)
    np.save(os.path.join(OUTPUT_DIR, 'metadata_filenames.npy'), metadata_filenames)
    print(f"Data and metadata saved to {OUTPUT_DIR}")

    return X

def visualize_sample(X_data):
    if len(X_data) == 0:
        print("No data to visualize.")
        return
    print("\nVisualizing one sample for a sanity check...")
    sample_data = X_data[0]
    
    # Dynamically create subplots based on the number of channels
    num_channels = sample_data.shape[2]
    fig, axs = plt.subplots(1, num_channels, figsize=(5 * num_channels, 4))
    if num_channels == 1:
        axs = [axs] # Make it iterable if there's only one
        
    fig.suptitle(f'Spectrograms for a Single Chunk ({num_channels} Mics)')
    
    for i in range(num_channels):
        ax = axs[i]
        actual_mic_index = MIC_CHANNELS_TO_USE[i]
        img = librosa.display.specshow(sample_data[:, :, i], sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title(f'Selected Mic Channel {actual_mic_index}')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    processed_X = process_data()
    visualize_sample(processed_X)