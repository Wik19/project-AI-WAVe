import os
import numpy as np
import tensorflow as tf
import librosa
import argparse
from collections import Counter

# --- Configuration ---
# IMPORTANT: These parameters must be IDENTICAL to the ones used for training.
MODEL_PATH = 'saved_models/uav_fault_model.keras' # Assuming you renamed it
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
CHUNK_SECONDS = 4
SAMPLES_PER_CHUNK = SAMPLE_RATE * CHUNK_SECONDS

# Map the output index back to a human-readable label
CLASS_NAMES = ['Healthy', 'Fractured', 'Edge-Distorted']

def predict_on_file(model, file_path):
    """
    Loads a single audio file, processes it into chunks, and predicts the
    health status for each rotor using a majority vote over the chunks.
    """
    print(f"\n--- Processing audio file: {os.path.basename(file_path)} ---")
    
    # --- Step 1: Process the audio file into spectrogram chunks ---
    spectrogram_chunks = []
    try:
        y_audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=False)
        
        # Check if the audio is long enough
        if y_audio.shape[1] < SAMPLES_PER_CHUNK:
            print(f"Error: Audio file is shorter than {CHUNK_SECONDS} seconds and cannot be processed.")
            return

        # Helper to convert a raw audio chunk to a spectrogram
        def chunk_to_spec(chunk):
            all_channels_mel = []
            for channel in range(chunk.shape[0]):
                mel_spec = librosa.feature.melspectrogram(
                    y=chunk[channel], sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
                )
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                all_channels_mel.append(log_mel_spec)
            return np.stack(all_channels_mel, axis=-1)

        # Create chunks from the audio file
        for i in range(0, y_audio.shape[1] - SAMPLES_PER_CHUNK + 1, SAMPLES_PER_CHUNK):
            chunk = y_audio[:, i : i + SAMPLES_PER_CHUNK]
            spectrogram_chunks.append(chunk_to_spec(chunk))

        # Convert list of chunks to a single NumPy array for the model
        X_predict = np.array(spectrogram_chunks)
        print(f"File successfully split into {len(X_predict)} chunks for analysis.")

    except Exception as e:
        print(f"Error processing audio file: {e}")
        return

    # --- Step 2: Make Predictions with the Model ---
    print("Model is making a prediction...")
    # The 'predictions' object will be a list of 4 arrays (one for each rotor)
    # Each array has shape (num_chunks, num_classes)
    predictions = model.predict(X_predict)
    
    # --- Step 3: Interpret the Predictions (Majority Vote) ---
    print("\n--- DIAGNOSIS RESULTS ---")
    for i in range(4):
        # For each rotor, get the predicted class index for every chunk
        # e.g., [0, 0, 1, 0, 0, 2, 0, ...]
        chunk_predictions = np.argmax(predictions[i], axis=1)
        
        # Find the most common prediction across all chunks (majority vote)
        # e.g., Counter({0: 5, 1: 1, 2: 1}).most_common(1) -> [(0, 5)]
        most_common_pred_index, count = Counter(chunk_predictions).most_common(1)[0]
        
        # Get the human-readable class name
        final_prediction = CLASS_NAMES[most_common_pred_index]
        
        # Calculate confidence
        confidence = (count / len(chunk_predictions)) * 100
        
        print(f"Rotor {i+1} Status: {final_prediction: <16} (Confidence: {confidence:.2f}%)")

def main():
    # --- Setup Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Diagnose UAV rotor faults from a WAV audio file.")
    parser.add_argument('-f', '--file', type=str, required=True, help="Path to the 4-channel WAV file to analyze.")
    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please ensure the model is trained and saved.")
        return
        
    if not os.path.exists(args.file):
        print(f"Error: Audio file not found at '{args.file}'. Please check the path.")
        return

    # --- Load Model ---
    print(f"Loading trained model from {MODEL_PATH}...")
    # We add compile=False because we are only doing inference, which is faster.
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # --- Run Prediction ---
    predict_on_file(model, args.file)

if __name__ == '__main__':
    main()