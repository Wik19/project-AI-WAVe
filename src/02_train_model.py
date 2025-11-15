# File: src/02_train_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# --- Configuration ---
PROCESSED_DATA_DIR = 'processed_data/'
MODEL_SAVE_DIR = 'saved_models/'

# Create the output directory if it doesn't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def load_data():
    """Loads the preprocessed data from disk."""
    print("Loading preprocessed data...")
    X = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_data.npy'))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_labels.npy'))
    print("Data loaded successfully.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def build_multi_head_cnn(input_shape, num_classes=3):
    """Builds a multi-head 2D CNN model."""
    inputs = layers.Input(shape=input_shape)
    
    # Common Feature Extractor
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    # Output Heads
    output_heads = []
    head_names = ['rotor1', 'rotor2', 'rotor3', 'rotor4']
    for i in range(4):
        head = layers.Dense(64, activation='relu')(x)
        head = layers.Dropout(0.3)(head)
        output = layers.Dense(num_classes, activation='softmax', name=head_names[i])(head)
        output_heads.append(output)

    model = models.Model(inputs=inputs, outputs=output_heads, name="uav_fault_cnn")
    return model

def plot_training_history(history):
    """Plots the training and validation history in a 3x2 grid."""
    head_names = ['rotor1', 'rotor2', 'rotor3', 'rotor4']
    
    # Create a 3x2 grid of plots
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Model Training History', fontsize=16)

    # --- Calculate Overall Metrics ---
    val_loss = np.sum([history.history[f'val_{name}_loss'] for name in head_names], axis=0)
    loss = np.sum([history.history[f'{name}_loss'] for name in head_names], axis=0)
    val_acc = np.mean([history.history[f'val_{name}_accuracy'] for name in head_names], axis=0)
    acc = np.mean([history.history[f'{name}_accuracy'] for name in head_names], axis=0)

    # --- Plot Overall Loss and Accuracy ---
    axs[0, 0].plot(loss, label='Training Loss')
    axs[0, 0].plot(val_loss, label='Validation Loss')
    axs[0, 0].set_title('Total Loss'); axs[0, 0].legend(); axs[0, 0].grid(True)
    
    axs[0, 1].plot(acc, label='Training Accuracy')
    axs[0, 1].plot(val_acc, label='Validation Accuracy')
    axs[0, 1].set_title('Average Accuracy'); axs[0, 1].legend(); axs[0, 1].grid(True)

    # --- Plot Individual Rotor Accuracies ---
    for i in range(4):
        # Correctly calculate the row and column for the 3x2 grid
        row = 1 + (i // 2)
        col = i % 2
        ax = axs[row, col]
        
        ax.plot(history.history[f'{head_names[i]}_accuracy'], label=f'Train Acc')
        ax.plot(history.history[f'val_{head_names[i]}_accuracy'], label=f'Val Acc')
        ax.set_title(f'Rotor {i+1} Accuracy'); ax.legend(); ax.grid(True)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
    
    # Hide the unused subplot if there is one
    if len(head_names) < 4:
       fig.delaxes(axs[2,1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    """Main function to run the training pipeline."""
    X, y = load_data()

    # Prepare labels for multi-task output
    y_prepared = [tf.keras.utils.to_categorical(y[:, i], num_classes=3) for i in range(4)]

    # Split data into training, validation, and test sets
    X_train, X_temp, y_train_temp, y_test_temp = train_test_split(X, list(zip(*y_prepared)), test_size=0.3, random_state=42)
    X_val, X_test, y_val_temp, y_test_temp = train_test_split(X_temp, y_test_temp, test_size=0.5, random_state=42)
    
    # Unzip the list of tuples back into lists of arrays
    y_train = [np.array(arr) for arr in zip(*y_train_temp)]
    y_val = [np.array(arr) for arr in zip(*y_val_temp)]
    y_test = [np.array(arr) for arr in zip(*y_test_temp)]

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    # Build and compile the model
    input_shape = X_train.shape[1:]
    model = build_multi_head_cnn(input_shape)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(  optimizer='adam', 
                    loss='categorical_crossentropy', # Keras applies this loss to all outputs
                    metrics={
                        'rotor1': 'accuracy',
                        'rotor2': 'accuracy',
                        'rotor3': 'accuracy',
                        'rotor4': 'accuracy'
                })
    model.summary()

    # Train the model
    EPOCHS = 30
    BATCH_SIZE = 32
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)] # Added for better training
    )

    # Save the trained model
    model.save(os.path.join(MODEL_SAVE_DIR, 'uav_fault_model.keras'))
    print(f"Model saved to {MODEL_SAVE_DIR}")

    # Evaluate on the test set
    test_loss, _, _, _, _, test_acc1, test_acc2, test_acc3, test_acc4 = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy for Rotor 1: {test_acc1:.4f}")
    print(f"Test Accuracy for Rotor 2: {test_acc2:.4f}")
    print(f"Test Accuracy for Rotor 3: {test_acc3:.4f}")
    print(f"Test Accuracy for Rotor 4: {test_acc4:.4f}")
    print(f"Average Test Accuracy: {np.mean([test_acc1, test_acc2, test_acc3, test_acc4]):.4f}")


    # Plot training history
    plot_training_history(history)

if __name__ == '__main__':
    main()