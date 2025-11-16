import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam                            # type: ignore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear      # type: ignore
from tf_keras_vis.utils.scores import CategoricalScore              # type: ignore
from sklearn.model_selection import train_test_split

# --- Configuration ---
PROCESSED_DATA_DIR = 'processed_data/'
MODEL_PATH = 'saved_models/uav_fault_model.keras'

FILENAME_TO_VISUALIZE = 'data/32_HEHF.wav' 
CHUNK_NUMBER_FROM_FILE = 10

def load_data():
    """Loads the preprocessed data and metadata from disk."""
    print("Loading preprocessed data...")
    X = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_data.npy'))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_labels.npy'))
    metadata = np.load(os.path.join(PROCESSED_DATA_DIR, 'metadata_filenames.npy'), allow_pickle=True)
    return X, y, metadata

def visualize_gradcam(original_model, x_sample, y_true_labels, sample_index):
    # ... (This function remains unchanged from the last correct version) ...
    class_names = ['Healthy', 'Fractured', 'Edge-Distorted']
    x_batch = np.expand_dims(x_sample, axis=0)
    last_conv_layer_name = None
    for layer in reversed(original_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
    if last_conv_layer_name is None:
        raise ValueError("Could not find a Conv2D layer in the model.")
    print(f"Generating Grad-CAM with respect to layer: {last_conv_layer_name}")
    fig, axs = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(f"Grad-CAM Analysis for Sample Index {sample_index}\n(File: {FILENAME_TO_VISUALIZE}, Chunk: {CHUNK_NUMBER_FROM_FILE})", fontsize=20)
    for i in range(4):
        output_tensor = original_model.outputs[i]
        single_output_model = tf.keras.models.Model(inputs=original_model.inputs, outputs=output_tensor)
        true_class_index = y_true_labels[i]
        score = CategoricalScore([true_class_index])
        gradcam = Gradcam(single_output_model, model_modifier=ReplaceToLinear(), clone=False)
        cam = gradcam(score, x_batch, penultimate_layer=last_conv_layer_name)
        heatmap = np.uint8(plt.cm.jet(cam[0])[..., :3] * 255)
        axs[i, 0].imshow(x_sample[:, :, i], cmap='viridis')
        axs[i, 0].set_title(f"Rotor {i+1}: Original Spectrogram")
        axs[i, 0].set_ylabel(f"True Label: {class_names[true_class_index]}")
        axs[i, 1].imshow(heatmap)
        axs[i, 1].set_title(f"Rotor {i+1}: Grad-CAM Heatmap")
        axs[i, 2].imshow(x_sample[:, :, i], cmap='viridis')
        axs[i, 2].imshow(heatmap, alpha=0.5)
        axs[i, 2].set_title(f"Rotor {i+1}: Overlay")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

def main():
    # 1. Load Data
    X, y, metadata = load_data()
    
    # --- THIS IS THE FIX ---
    # 2. Split data using a 1D array for stratification
    # We will stratify based on the labels of the first rotor as a proxy
    y_stratify = y[:, 0]
    
    # We must pass all arrays we want to split. The function returns them in the same order.
    # To avoid unused variables, we can use a single variable for the "train" part.
    split_arrays = train_test_split(
        X, y, metadata, test_size=0.3, random_state=42, stratify=y_stratify
    )
    X_train, X_temp, y_train_raw, y_temp_raw, metadata_train, metadata_temp = split_arrays

    # Stratify the second split as well
    y_temp_stratify = y_temp_raw[:, 0]
    split_arrays_2 = train_test_split(
        X_temp, y_temp_raw, metadata_temp, test_size=0.5, random_state=42, stratify=y_temp_stratify
    )
    X_val, X_test, y_val_raw, y_test_raw, metadata_val, metadata_test = split_arrays_2
    
    # 3. Load the Champion Model
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        return
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 4. Find the index of the specific sample
    try:
        indices_from_file = np.where(metadata_test == FILENAME_TO_VISUALIZE)[0]
        if len(indices_from_file) == 0:
            print(f"File '{FILENAME_TO_VISUALIZE}' was not found in the generated test set.")
            return
        if CHUNK_NUMBER_FROM_FILE >= len(indices_from_file):
            print(f"You requested chunk #{CHUNK_NUMBER_FROM_FILE}, but only {len(indices_from_file)} chunks from this file exist in the test set.")
            return
        sample_index_in_test_set = indices_from_file[CHUNK_NUMBER_FROM_FILE]
        print(f"Found '{FILENAME_TO_VISUALIZE}' (chunk #{CHUNK_NUMBER_FROM_FILE}) at index {sample_index_in_test_set} in the test set.")
    except Exception as e:
        print(f"An error occurred while finding the sample: {e}")
        return

    # 5. Extract the data for that specific index
    sample_to_explain = X_test[sample_index_in_test_set]
    true_labels_for_sample = y_test_raw[sample_index_in_test_set]
    
    # 6. Run the visualization
    visualize_gradcam(model, sample_to_explain, true_labels_for_sample, sample_index_in_test_set)

if __name__ == '__main__':
    main()