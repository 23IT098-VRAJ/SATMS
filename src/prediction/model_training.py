import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from traffic_predictor_tf import TrafficPredictor
from data_preprocessing import TrafficDataProcessor

def generate_synthetic_traffic_data(n_samples=1000):
    """Generate synthetic traffic data for testing when real data is unavailable"""
    print("Generating synthetic traffic data...")
    
    # Create time index
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='15min')
    
    # Initialize dataframe
    data = pd.DataFrame({'timestamp': timestamps})
    
    # Add time-based features
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    data['month'] = data['timestamp'].dt.month
    
    # Create base traffic patterns
    # Morning rush hour (7-9 AM)
    morning_rush = ((data['hour'] >= 7) & (data['hour'] <= 9)).astype(float) * 0.6
    # Evening rush hour (4-6 PM)
    evening_rush = ((data['hour'] >= 16) & (data['hour'] <= 18)).astype(float) * 0.7
    # Weekend reduction
    weekend_factor = 1.0 - (data['is_weekend'] * 0.3)
    
    # Generate traffic density with realistic patterns
    base_density = np.random.normal(0.4, 0.1, size=n_samples)
    data['traffic_density'] = (base_density + morning_rush + evening_rush) * weekend_factor
    data['traffic_density'] = np.clip(data['traffic_density'], 0.1, 1.0)
    
    # Generate speed (inversely related to density)
    data['average_speed'] = 60 - (data['traffic_density'] * 40) + np.random.normal(0, 5, size=n_samples)
    data['average_speed'] = np.clip(data['average_speed'], 5, 60)
    
    # Generate flow rate (density * speed * road capacity factor)
    data['flow_rate'] = data['traffic_density'] * data['average_speed'] * 0.8 + np.random.normal(0, 3, size=n_samples)
    data['flow_rate'] = np.clip(data['flow_rate'], 0, 100)
    
    # Add some sensor data
    data['vehicle_count'] = (data['traffic_density'] * 100 + np.random.normal(0, 10, size=n_samples)).astype(int)
    data['vehicle_count'] = np.clip(data['vehicle_count'], 0, 200)
    
    # Add zone information
    zones = ['zone_1', 'zone_2', 'zone_3', 'zone_4']
    for zone in zones:
        zone_factor = 0.8 + (hash(zone) % 5) / 10.0  # Zone-specific factor between 0.8-1.2
        data[f'{zone}_density'] = data['traffic_density'] * zone_factor + np.random.normal(0, 0.05, size=n_samples)
        data[f'{zone}_density'] = np.clip(data[f'{zone}_density'], 0.1, 1.0)
    
    return data

def augment_traffic_data(X, y, augmentation_factor=0.2):
    """Augment traffic data with noise and time shifts"""
    X_augmented = []
    y_augmented = []
    
    # Original data
    X_augmented.append(X)
    y_augmented.append(y)
    
    # Add noise
    noise = np.random.normal(0, augmentation_factor, X.shape)
    X_noise = X + noise
    X_augmented.append(X_noise)
    y_augmented.append(y)
    
    # Time shift (forward)
    if X.shape[1] > 2:  # If we have enough time steps
        X_shift_forward = X[:, 1:, :]
        # Repeat the last time step
        last_step = X[:, -1:, :]
        X_shift_forward = np.concatenate([X_shift_forward, last_step], axis=1)
        X_augmented.append(X_shift_forward)
        y_augmented.append(y)
    
    # Time shift (backward)
    if X.shape[1] > 2:
        X_shift_backward = X[:, :-1, :]
        # Repeat the first time step
        first_step = X[:, :1, :]
        X_shift_backward = np.concatenate([first_step, X_shift_backward], axis=1)
        X_augmented.append(X_shift_backward)
        y_augmented.append(y)
    
    # Combine all augmented data
    X_combined = np.vstack(X_augmented)
    y_combined = np.vstack(y_augmented) if len(y.shape) > 1 else np.hstack(y_augmented)
    
    return X_combined, y_combined

def evaluate_and_visualize_model(predictor, X_test, y_test, save_dir):
    """Evaluate model and create visualizations"""
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Make predictions
    y_pred = predictor.predict(X_test)
    
    # Convert predictions to the right format for evaluation
    if isinstance(y_pred, list) and isinstance(y_pred[0], dict):
        # Extract values from prediction dictionaries
        # Assuming consistent structure across all predictions
        first_zone = list(y_pred[0].keys())[0]
        if isinstance(y_pred[0][first_zone], dict):
            # Multi-output case
            metrics_keys = list(y_pred[0][first_zone].keys())
            y_pred_array = np.zeros((len(y_pred), len(metrics_keys)))
            for i, pred_dict in enumerate(y_pred):
                for j, metric in enumerate(metrics_keys):
                    y_pred_array[i, j] = pred_dict[first_zone][metric]
            y_pred = y_pred_array
        else:
            # Single output case
            y_pred_array = np.array([pred[first_zone] for pred in y_pred]).reshape(-1, 1)
            y_pred = y_pred_array
    
    # For multi-output case
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        metrics = {}
        fig, axes = plt.subplots(y_test.shape[1], 1, figsize=(12, 4*y_test.shape[1]))
        
        for i in range(y_test.shape[1]):
            y_true_i = y_test[:, i]
            y_pred_i = y_pred[:, i] if len(y_pred.shape) > 1 else y_pred
            
            # Calculate metrics
            mse = mean_squared_error(y_true_i, y_pred_i)
            mae = mean_absolute_error(y_true_i, y_pred_i)
            r2 = r2_score(y_true_i, y_pred_i)
            
            metrics[f'output_{i}'] = {'mse': mse, 'mae': mae, 'r2': r2}
            
            # Plot actual vs predicted
            ax = axes[i] if y_test.shape[1] > 1 else axes
            ax.plot(y_true_i, label='Actual')
            ax.plot(y_pred_i, label='Predicted')
            ax.set_title(f'Output {i}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}')
            ax.legend()
            ax.grid(True)
    else:
        # For single output
        y_pred_flat = y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred_flat)
        mae = mean_absolute_error(y_test, y_pred_flat)
        r2 = r2_score(y_test, y_pred_flat)
        
        metrics = {'mse': mse, 'mae': mae, 'r2': r2}
        
        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual')
        plt.plot(y_pred_flat, label='Predicted')
        plt.title(f'Traffic Prediction: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}')
        plt.legend()
        plt.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_results.png'))
    
    # Save metrics
    with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def save_training_summary(model_path, fold_metrics, history):
    """Save training summary and metrics"""
    summary_dir = os.path.dirname(model_path)
    os.makedirs(summary_dir, exist_ok=True)
    
    # Save fold metrics
    with open(os.path.join(summary_dir, 'fold_metrics.json'), 'w') as f:
        json.dump(fold_metrics, f, indent=4)
    
    # Save training history
    if hasattr(history, 'history'):
        history_dict = history.history
        with open(os.path.join(summary_dir, 'training_history.json'), 'w') as f:
            # Convert numpy values to floats for JSON serialization
            serializable_history = {}
            for key, values in history_dict.items():
                serializable_history[key] = [float(val) for val in values]
            json.dump(serializable_history, f, indent=4)
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Training and Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, 'training_history.png'))

def train_traffic_model(data_path, model_save_path, time_steps=10, test_size=0.2, multi_output=True):
    """Train a traffic prediction model with historical data
    
    Args:
        data_path: Path to CSV file with historical traffic data
        model_save_path: Path to save the trained model
        time_steps: Number of time steps for sequence prediction
        test_size: Proportion of data to use for testing
        multi_output: Whether to predict multiple traffic metrics
    
    Returns:
        Trained model and evaluation metrics
    """
    # Create output directories
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Load historical data
    try:
        print(f"Loading data from {data_path}...")
        data = pd.read_csv(data_path)
        print(f"Loaded {len(data)} records from {data_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback: generate synthetic data for testing
        print("Generating synthetic data for testing...")
        data = generate_synthetic_traffic_data(1000)
    
    # Initialize data processor
    processor = TrafficDataProcessor()
    
    # Prepare features and targets (with multi-output support)
    X, y = processor.prepare_features(data, time_steps=time_steps, multi_output=multi_output)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # Augment training data
    print("Augmenting training data...")
    X_train_aug, y_train_aug = augment_traffic_data(X_train, y_train, augmentation_factor=0.1)
    print(f"Training data augmented from {len(X_train)} to {len(X_train_aug)} samples")
    
    # Initialize and build model
    predictor = TrafficPredictor()
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
    predictor.build_model(input_shape, output_size)
    
    # Train model with cross-validation
    kf = KFold(n_splits=5)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_aug)):
        print(f"Training fold {fold+1}/5...")
        X_fold_train, X_fold_val = X_train_aug[train_idx], X_train_aug[val_idx]
        y_fold_train, y_fold_val = y_train_aug[train_idx], y_train_aug[val_idx]
        
        history = predictor.train(
            X_fold_train, y_fold_train,
            X_fold_val, y_fold_val,
            epochs=50,  # Reduced for faster training in each fold
            batch_size=32,
            save_path=f"{model_save_path}.fold{fold}"
        )
        
        # Evaluate on validation set
        y_pred = predictor.predict(X_fold_val)
        
        # Calculate metrics (handling multi-output case)
        if multi_output and len(y_test.shape) > 1:
            metrics = {}
            for i in range(y_test.shape[1]):
                mse = np.mean((y_fold_val[:, i] - y_pred[:, i]) ** 2)
                mae = np.mean(np.abs(y_fold_val[:, i] - y_pred[:, i]))
                metrics[f'output_{i}_mse'] = float(mse)
                metrics[f'output_{i}_mae'] = float(mae)
        else:
            y_pred_flat = y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
            mse = np.mean((y_fold_val - y_pred_flat) ** 2)
            mae = np.mean(np.abs(y_fold_val - y_pred_flat))
            metrics = {'mse': float(mse), 'mae': float(mae)}
        
        fold_metrics.append(metrics)
        print(f"Fold {fold+1} metrics: {metrics}")
    
    # Train final model on all training data
    print("\nTraining final model on all training data...")
    history = predictor.train(
        X_train_aug, y_train_aug,
        X_test, y_test,
        epochs=100,
        batch_size=32,
        save_path=model_save_path
    )
    
    # Evaluate and visualize final model
    print("\nEvaluating final model...")
    eval_dir = os.path.join(os.path.dirname(model_save_path), "evaluation")
    metrics = evaluate_and_visualize_model(predictor, X_test, y_test, eval_dir)
    
    # Save training summary
    save_training_summary(model_save_path, fold_metrics, history)
    
    print(f"Model training complete. Model saved to {model_save_path}")
    print(f"Evaluation results saved to {eval_dir}")
    
    return predictor, {
        'metrics': metrics, 
        'history': history, 
        'fold_metrics': fold_metrics
    }

if __name__ == "__main__":
    # Example usage
    train_traffic_model(
        data_path="training_data/historical_traffic_data.csv",
        model_save_path="models/saved_models/traffic_lstm_model.h5",
        time_steps=12,
        multi_output=True
    )
