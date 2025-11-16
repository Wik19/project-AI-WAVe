import os
import numpy as np                                                      # type: ignore
import tensorflow as tf                                                 # type: ignore
from sklearn.metrics import confusion_matrix, classification_report     # type: ignore
import seaborn as sns                                                   # type: ignore
import matplotlib.pyplot as plt                                         # type: ignore
from sklearn.model_selection import train_test_split                    # type: ignore

# --- Configuration ---
PROCESSED_DATA_DIR = 'processed_data/'
MODEL_PATH = 'saved_models/uav_fault_model.keras'

def load_data():
    print("Loading preprocessed data...")
    X = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_data.npy'))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_labels.npy'))
    return X, y

def plot_confusion_matrices(model, X_test, y_test):
    # Get predictions from the model
    # predictions will be a list of 4 arrays (one for each rotor)
    predictions = model.predict(X_test)
    
    class_names = ['Healthy', 'Fractured', 'Edge-Distorted']
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confusion Matrices per Rotor', fontsize=16)
    
    for i in range(4):
        ax = axs[i // 2, i % 2]
        
        # Get true labels for this rotor (convert from one-hot if necessary, or just take index)
        # y_test[i] is the one-hot encoded array for rotor i
        y_true_classes = np.argmax(y_test[i], axis=1)
        
        # Get predicted classes
        y_pred_classes = np.argmax(predictions[i], axis=1)
        
        # Compute matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f'Rotor {i+1}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # Print a text report as well
        print(f"\n--- Classification Report for Rotor {i+1} ---")
        print(classification_report(y_true_classes, y_pred_classes, target_names=class_names, zero_division=0))

    plt.tight_layout()
    plt.show()

def main():
    # 1. Load Data
    X, y = load_data()
    
    # 2. Prepare labels exactly like we did in training
    y_prepared = [tf.keras.utils.to_categorical(y[:, i], num_classes=3) for i in range(4)]
    
    # 3. Split to get the same Test set as before
    # Note: We must use the same random_state to ensure we get the same "Test" set
    X_train, X_temp, y_train_temp, y_test_temp = train_test_split(
        X, list(zip(*y_prepared)), test_size=0.3, random_state=42
    )
    X_val, X_test, y_val_temp, y_test_temp_final = train_test_split(
        X_temp, y_test_temp, test_size=0.5, random_state=42
    )
    
    # Unzip y_test
    y_test = [np.array(arr) for arr in zip(*y_test_temp_final)]
    
    # 4. Load Model
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Please train the model first.")
        return
        
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 5. Evaluate
    plot_confusion_matrices(model, X_test, y_test)

if __name__ == '__main__':
    main()