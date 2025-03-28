import tensorflow as tf
import numpy as np
import os

print("Creating a basic traffic prediction model...")

# Create a simple LSTM model for traffic prediction
def create_basic_model():
    # Define input shape (time steps, features)
    time_steps = 10
    features = 5
    
    # Create a simple LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(time_steps, features), return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Create some dummy data for testing
    X_dummy = np.random.random((100, time_steps, features))
    y_dummy = np.random.random(100)
    
    # Train for a few epochs just to initialize weights
    model.fit(X_dummy, y_dummy, epochs=2, verbose=1, batch_size=16)
    
    return model

# Ensure the model directory exists
os.makedirs('models/saved_models', exist_ok=True)

# Create and save the model
model = create_basic_model()
model_path = 'models/saved_models/traffic_lstm_model.h5'
model.save(model_path)

print(f"Basic model created and saved to {model_path}")
print("You can now run the main application.")