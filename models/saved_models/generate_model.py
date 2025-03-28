# generate_model.py
import tensorflow as tf
import numpy as np
import os

print("Creating an enhanced traffic prediction model...")

# Create a multi-output LSTM model for traffic prediction
def create_enhanced_model():
    # Define input shape (time steps, features)
    time_steps = 12
    features = 8
    output_size = 3  # For density, speed, flow rate
    
    # Create an enhanced LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(time_steps, features), return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_size)
    ])
    
    # Compile the model with improved settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    # Create some dummy data for initialization
    X_dummy = np.random.random((100, time_steps, features))
    y_dummy = np.random.random((100, output_size))
    
    # Train for a few epochs just to initialize weights
    model.fit(X_dummy, y_dummy, epochs=2, verbose=1, batch_size=16)
    
    return model

# Ensure the model directory exists
os.makedirs('models/saved_models', exist_ok=True)

# Create and save the model
model = create_enhanced_model()
model_path = 'models/saved_models/traffic_lstm_model.h5'
model.save(model_path)

# Save model summary to a text file for reference
model_summary_path = 'models/saved_models/model_summary.txt'
with open(model_summary_path, 'w') as f:
    # Redirect summary to file
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print(f"Enhanced model created and saved to {model_path}")
print(f"Model summary saved to {model_summary_path}")
print("You can now run the main application.")

# Create a metadata file
metadata_path = 'models/saved_models/model_metadata.json'
import json
with open(metadata_path, 'w') as f:
    json.dump({
        "time_steps": 12,
        "features": 8,
        "output_size": 3,
        "metrics": ["density", "speed", "flow_rate"],
        "created_at": tf.timestamp().numpy().tolist(),
        "tensorflow_version": tf.__version__
    }, f, indent=4)
print(f"Model metadata saved to {metadata_path}")