import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import os

class TrafficPredictor:
    """Predict traffic patterns using LSTM neural network"""
    
    def __init__(self, model_path=None):
        """Initialize the traffic predictor
        
        Args:
            model_path: Path to saved model (if loading existing model)
        """
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            self.model = None
            
    def build_model(self, input_shape, output_size=1):
        """Build LSTM model for traffic prediction
        
        Args:
            input_shape: Shape of input data (time_steps, features)
            output_size: Number of output values to predict
        """
        model = Sequential([
            LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(output_size)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
        
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, save_path=None):
        """Train the traffic prediction model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            save_path: Path to save trained model
            
        Returns:
            Training history
        """
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
            
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Model checkpoint to save best model
        if save_path:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                save_path,
                monitor='val_loss',
                save_best_only=True
            )
            callbacks = [early_stopping, checkpoint]
        else:
            callbacks = [early_stopping]
            
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Save final model if path provided
        if save_path and not os.path.exists(save_path):
            self.model.save(save_path)
            
        return history
        
    def predict(self, X):
        """Make traffic predictions
        
        Args:
            X: Input features
            
        Returns:
            Predicted traffic values
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() or load a saved model.")
            
        return self.model.predict(X)
        
    def save_model(self, path):
        """Save the model to disk
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        self.model.save(path)
        
    def load_model(self, path):
        """Load model from disk
        
        Args:
            path: Path to the saved model
        """
        if os.path.exists(path):
            self.model = load_model(path)
        else:
            raise FileNotFoundError(f"No model found at {path}")