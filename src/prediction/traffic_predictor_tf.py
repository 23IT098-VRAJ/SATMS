import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import numpy as np
import os
import json
from datetime import datetime

class TrafficPredictor:
    """Predict traffic patterns using LSTM neural network"""
    
    def __init__(self, model_path=None):
        """Initialize the traffic predictor
        
        Args:
            model_path: Path to saved model (if loading existing model)
        """
        self.model = None
        self.output_size = 1
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def build_model(self, input_shape, output_size=3):
        """Build LSTM model for traffic prediction
        
        Args:
            input_shape: Shape of input data (time_steps, features)
            output_size: Number of output values to predict
        """
        self.output_size = output_size
        
        model = Sequential([
            LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True, name="lstm_1"),
            BatchNormalization(name="batch_norm_1"),
            Dropout(0.3, name="dropout_1"),
            LSTM(64, activation='relu', name="lstm_2"),
            BatchNormalization(name="batch_norm_2"),
            Dropout(0.3, name="dropout_2"),
            Dense(32, activation='relu', name="dense_1"),
            Dense(output_size, name="output")
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
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
            output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
            self.build_model(input_shape, output_size)
        
        # Create log directory for TensorBoard
        log_dir = "monitoring/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, 
            histogram_freq=1,
            write_graph=True
        )
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Learning rate scheduler
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
        
        # Model checkpoint to save best model
        callbacks = [early_stopping, lr_scheduler, tensorboard_callback]
        
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model if path provided and not already saved by checkpoint
        if save_path and not os.path.exists(save_path):
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
        
        # Save model summary
        if save_path:
            summary_path = os.path.splitext(save_path)[0] + "_summary.txt"
            with open(summary_path, 'w') as f:
                # Save model architecture
                self.model.summary(print_fn=lambda x: f.write(x + '\n'))
                
                # Save training parameters
                f.write("\nTraining Parameters:\n")
                f.write(f"Epochs: {epochs}\n")
                f.write(f"Batch Size: {batch_size}\n")
                f.write(f"Training Samples: {len(X_train)}\n")
                f.write(f"Validation Samples: {len(X_val)}\n")
                
                # Save final metrics
                f.write("\nFinal Metrics:\n")
                for metric, value in zip(self.model.metrics_names, self.model.evaluate(X_val, y_val, verbose=0)):
                    f.write(f"{metric}: {value:.4f}\n")
        
        return history
    
    def predict(self, X):
        """Make traffic predictions
        
        Args:
            X: Input features
            
        Returns:
            Predicted traffic values formatted for controller
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() or load a saved model.")
        
        # Get raw predictions from the model
        raw_predictions = self.model.predict(X)
        
        # Format predictions as expected by the controller
        predictions_list = []
        
        for i in range(len(raw_predictions)):
            prediction_dict = {}
            
            # Create predictions for each zone
            for zone_id in ['zone_1', 'zone_2', 'zone_3', 'zone_4']:
                # If we have multi-output predictions (output_size > 1)
                if self.output_size > 1:
                    # Create a nested dictionary with multiple metrics
                    prediction_dict[zone_id] = {
                        'density': float(raw_predictions[i][0]),
                        'speed': float(raw_predictions[i][1]),
                        'flow_rate': float(raw_predictions[i][2]) if self.output_size > 2 else float(raw_predictions[i][0] * raw_predictions[i][1] * 0.1)
                    }
                else:
                    # Single output case
                    prediction_dict[zone_id] = float(raw_predictions[i][0])
            
            predictions_list.append(prediction_dict)
        
        return predictions_list
    
    def save_model(self, path):
        """Save the model to disk
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        self.model.save(path)
        print(f"Model saved to {path}")
        
        # Save metadata
        metadata = {
            "output_size": self.output_size,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tensorflow_version": tf.__version__
        }
        
        metadata_path = os.path.splitext(path)[0] + "_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def load_model(self, path):
        """Load model from disk
        
        Args:
            path: Path to the saved model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        
        # Load the model
        self.model = load_model(path)
        print(f"Model loaded from {path}")
        
        # Try to load metadata if available
        metadata_path = os.path.splitext(path)[0] + "_metadata.json"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.output_size = metadata.get("output_size", 1)
                    print(f"Loaded model metadata: output_size={self.output_size}")
            except:
                # Infer output size from model
                self.output_size = self.model.layers[-1].output_shape[-1]
                print(f"Inferred output_size={self.output_size} from model architecture")
        else:
            # Infer output size from model
            self.output_size = self.model.layers[-1].output_shape[-1]
            print(f"Inferred output_size={self.output_size} from model architecture")
    
    def evaluate(self, X_test, y_test, detailed=False):
        """Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            detailed: Whether to return detailed metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() or load a saved model.")
        
        # Get basic metrics from model.evaluate
        base_metrics = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {name: float(value) for name, value in zip(self.model.metrics_names, base_metrics)}
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # If detailed evaluation requested
        if detailed:
            # For multi-output case
            if self.output_size > 1 and len(y_test.shape) > 1:
                for i in range(self.output_size):
                    output_name = f"output_{i}"
                    metrics[f"{output_name}_mse"] = float(np.mean((y_test[:, i] - y_pred[:, i]) ** 2))
                    metrics[f"{output_name}_mae"] = float(np.mean(np.abs(y_test[:, i] - y_pred[:, i])))
                    
                    # Calculate R²
                    y_mean = np.mean(y_test[:, i])
                    ss_total = np.sum((y_test[:, i] - y_mean) ** 2)
                    ss_residual = np.sum((y_test[:, i] - y_pred[:, i]) ** 2)
                    metrics[f"{output_name}_r2"] = float(1 - (ss_residual / ss_total)) if ss_total > 0 else 0
            
            # Add more detailed metrics as needed
            
            # Add prediction samples
            num_samples = min(5, len(y_test))
            samples = []
            
            for i in range(num_samples):
                sample = {
                    "actual": y_test[i].tolist() if isinstance(y_test[i], np.ndarray) else float(y_test[i]),
                    "predicted": y_pred[i].tolist() if isinstance(y_pred[i], np.ndarray) else float(y_pred[i])
                }
                samples.append(sample)
            
            metrics["samples"] = samples
        
        return metrics