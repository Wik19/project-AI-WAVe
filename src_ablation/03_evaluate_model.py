import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- CONFIGURATION: POINT THIS TO THE CORRECT DATA FOLDER ---
PROCESSED_DATA_DIR = 'processed_data_mics_02/'
MODEL_PATH = 'saved_models/uav_fault_model_02.keras'

def load_data():
    """Loads the preprocessed data from the specified directory."""
    print(f"Loading preprocessed data from: {PROCESSED_DATA_DIR}")
    X = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_data.npy'))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_labels.npy'))
    return X, y

def plot_confusion_matrices(model, X_test, y_test):
    """Generates and plots confusion matrices and classification reports."""
    predictions = model.predict(X_test)
    class_names = ['Healthy', 'Fractured', 'Edge-Distorted']
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Confusion Matrices for Model Evaluation\n(Data: {PROCESSED_DATA_DIR})', fontsize=16)
    
    for i in range(4):
        ax = axs[i // 2, i % 2]
        y_true_classes = np.argmax(y_test[i], axis=1)
        y_pred_classes = np.argmax(predictions[i], axis=1)
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f'Rotor {i+1}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        print(f"\n--- Classification Report for Rotor {i+1} ---")
        print(classification_report(y_true_classes, y_pred_classes, target_names=class_names, zero_division=0))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    # 1. Load Data
    X, y = load_data()
    
    # 2. Replicate the EXACT same data split used during training to get the correct test set.
    # We stratify using the labels of the first rotor as a proxy for class distribution.
    y_stratify = y[:, 0]
    
    # First split (70% train, 30% temp)
    # We only need the temporary sets, so we use underscores "_" to discard the training sets.
    _, X_temp, _, y_temp_raw = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y_stratify
    )

    # Second split of the temporary set (50% validation, 50% test)
    y_temp_stratify = y_temp_raw[:, 0]
    # We only need the test set, so we discard the validation set.
    _, X_test, _, y_test_raw = train_test_split(
        X_temp, y_temp_raw, test_size=0.5, random_state=42, stratify=y_temp_stratify
    )
    
    # 3. One-hot encode the final test labels for evaluation
    y_test = [tf.keras.utils.to_categorical(y_test_raw[:, i], num_classes=3) for i in range(4)]
    
    # 4. Load the trained model
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Make sure you have trained the corresponding model.")
        return
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 5. Evaluate and plot
    plot_confusion_matrices(model, X_test, y_test)

if __name__ == '__main__':
    main()