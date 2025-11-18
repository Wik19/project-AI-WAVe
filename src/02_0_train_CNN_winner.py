import numpy as np                                          # type: ignore
import tensorflow as tf                                     # type: ignore
from tensorflow.keras import layers, models                 # type: ignore
from sklearn.model_selection import train_test_split        # type: ignore
from sklearn.utils import class_weight                      # type: ignore
import matplotlib.pyplot as plt                             # type: ignore
import os

# --- Configuration ---
# PROCESSED_DATA_DIR = 'processed_data/'
PROCESSED_DATA_DIR = 'processed_data_mics_02/'
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
    """Builds a deeper, stable multi-head 2D CNN model."""
    inputs = layers.Input(shape=input_shape)

    # --- Common Feature Extractor Body ---
    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # NEW Block 4 - Gives the model more learning capacity
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten for the dense layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x) # Use 0.5 dropout for this deeper model

    # --- Output Heads ---
    output_heads = []
    head_names = ['rotor1', 'rotor2', 'rotor3', 'rotor4']

    for i in range(4):
        head = layers.Dense(128, activation='relu')(x) # Slightly larger dense layer
        head = layers.Dropout(0.5)(head)
        output = layers.Dense(num_classes, activation='softmax', name=head_names[i])(head)
        output_heads.append(output)

    model = models.Model(inputs=inputs, outputs=output_heads, name="uav_fault_deep_cnn")
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
    """Main function to run the training pipeline with SAMPLE weights for class imbalance."""
    X, y = load_data()

    # --- Step 1: Calculate class weights ---
    class_weights_per_output = []
    for i in range(4):
        weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y[:, i]),
            y=y[:, i]
        )
        class_weights_per_output.append(dict(enumerate(weights)))

    print("Calculated Class Weights:")
    for i, w in enumerate(class_weights_per_output):
        print(f"  Rotor {i+1}: {w}")

    # --- Step 2: Split the data ---
    X_train, X_temp, y_train_raw, y_temp_raw = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val_raw, y_test_raw = train_test_split(
        X_temp, y_temp_raw, test_size=0.5, random_state=42
    )

    # --- Step 3: Compute sample weights for the training set ---
    # We will build a list directly this time.
    sample_weights_list = []
    for i in range(4):
        rotor_class_weights = class_weights_per_output[i]
        rotor_sample_weights = np.array([rotor_class_weights[label] for label in y_train_raw[:, i]])
        sample_weights_list.append(rotor_sample_weights)
        
    print(f"\nCreated sample weights list for {len(X_train)} training samples.")

    # --- Step 4: One-hot encode the labels ---
    y_train = [tf.keras.utils.to_categorical(y_train_raw[:, i], num_classes=3) for i in range(4)]
    y_val = [tf.keras.utils.to_categorical(y_val_raw[:, i], num_classes=3) for i in range(4)]
    y_test = [tf.keras.utils.to_categorical(y_test_raw[:, i], num_classes=3) for i in range(4)]

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    # Build and compile model
    input_shape = X_train.shape[1:]
    model = build_multi_head_cnn(input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy',
                  metrics={'rotor1': 'accuracy', 'rotor2': 'accuracy', 'rotor3': 'accuracy', 'rotor4': 'accuracy'})
    model.summary()

    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

    # --- MODIFIED: TRAIN WITH A LIST of sample_weights ---
    EPOCHS = 50
    BATCH_SIZE = 32
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        sample_weight=sample_weights_list
    )

    # Save and evaluate (the rest of the script is the same)
    model.save(os.path.join(MODEL_SAVE_DIR, 'uav_fault_model.keras'))
    
    # ... evaluation logic ...
    print("\n--- Evaluating on Test Set ---")
    test_results = model.evaluate(X_test, y_test, return_dict=True)
    print(test_results)

    # ... plotting logic ...
    plot_training_history(history)

if __name__ == '__main__':
    main()