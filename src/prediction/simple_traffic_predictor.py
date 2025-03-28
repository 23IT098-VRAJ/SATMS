import numpy as np
import os
import json
from datetime import datetime

class SimpleTrafficPredictor:
    """A simplified traffic predictor that doesn't rely on TensorFlow"""
    
    def __init__(self, model_path=None):
        """Initialize the traffic predictor"""
        self.model = "SimpleModel"
        self.config = self._load_config()
        print("Using simplified traffic predictor (no TensorFlow)")
    
    def _load_config(self):
        """Load configuration or use defaults"""
        config_path = 'config/model_config.json'
        default_config = {
            "time_patterns": {
                "morning_rush_start": 7,
                "morning_rush_end": 9,
                "evening_rush_start": 16,
                "evening_rush_end": 18,
                "weekend_factor": 0.7
            },
            "zone_factors": {
                "zone_1": 1.0,
                "zone_2": 1.1,
                "zone_3": 0.9,
                "zone_4": 1.2
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults for any missing keys
                    if "time_patterns" not in config:
                        config["time_patterns"] = default_config["time_patterns"]
                    if "zone_factors" not in config:
                        config["zone_factors"] = default_config["zone_factors"]
                return config
            except:
                return default_config
        else:
            return default_config
    
    def build_model(self, input_shape, output_size=3):
        """Build a simple model for traffic prediction"""
        print(f"Building simple model with input shape {input_shape}")
        self.input_shape = input_shape
        self.output_size = output_size
        self.model = "SimpleModel"
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, save_path=None):
        """Train the traffic prediction model"""
        print("Simple model doesn't require training")
        # Store some statistics from the training data to use in predictions
        self.data_stats = {
            "mean": np.mean(X_train, axis=(0, 1)),
            "std": np.std(X_train, axis=(0, 1)),
            "min": np.min(X_train, axis=(0, 1)),
            "max": np.max(X_train, axis=(0, 1))
        }
        
        if len(y_train.shape) > 1:
            self.target_stats = {
                "mean": np.mean(y_train, axis=0),
                "std": np.std(y_train, axis=0),
                "min": np.min(y_train, axis=0),
                "max": np.max(y_train, axis=0)
            }
        else:
            self.target_stats = {
                "mean": np.mean(y_train),
                "std": np.std(y_train),
                "min": np.min(y_train),
                "max": np.max(y_train)
            }
        
        # Save these statistics if a path is provided
        if save_path:
            stats_path = save_path + ".stats.json"
            stats = {
                "data_stats": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.data_stats.items()},
                "target_stats": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.target_stats.items()}
            }
            
            try:
                os.makedirs(os.path.dirname(stats_path), exist_ok=True)
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=4)
                print(f"Model statistics saved to {stats_path}")
            except Exception as e:
                print(f"Could not save statistics: {e}")
        
        return {"loss": [0.1], "val_loss": [0.2]}
    
    def predict(self, X):
        """Make traffic predictions using a simple rule-based approach"""
        # Get current time for time-based patterns
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()  # 0-6 (Monday is 0)
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Time patterns from config
        time_patterns = self.config.get("time_patterns", {})
        morning_rush_start = time_patterns.get("morning_rush_start", 7)
        morning_rush_end = time_patterns.get("morning_rush_end", 9)
        evening_rush_start = time_patterns.get("evening_rush_start", 16)
        evening_rush_end = time_patterns.get("evening_rush_end", 18)
        weekend_factor = time_patterns.get("weekend_factor", 0.7)
        
        # Zone factors from config
        zone_factors = self.config.get("zone_factors", {
            "zone_1": 1.0,
            "zone_2": 1.1,
            "zone_3": 0.9,
            "zone_4": 1.2
        })
        
        # Handle different input types
        if isinstance(X, np.ndarray):
            if len(X.shape) == 3:  # Batch of sequences
                # Extract more meaningful features with weighted average
                # More weight on recent time steps
                weights = np.linspace(0.5, 1.0, X.shape[1])
                weighted_values = X * weights.reshape(1, -1, 1)
                predictions_base = np.sum(weighted_values, axis=1) / np.sum(weights)
                
                # Apply time-based factors
                time_factor = 1.0
                if hour >= morning_rush_start and hour <= morning_rush_end:
                    time_factor = 1.5  # Morning rush hour
                elif hour >= evening_rush_start and hour <= evening_rush_end:
                    time_factor = 1.6  # Evening rush hour (slightly heavier)
                
                # Weekend adjustment
                if is_weekend:
                    time_factor *= weekend_factor
                
                # Apply time factor to base predictions
                predictions = predictions_base * time_factor
                
                # Format as list of dictionaries for compatibility with controller
                predictions_list = []
                for i in range(len(predictions)):
                    prediction_dict = {}
                    
                    for zone_id, zone_factor in zone_factors.items():
                        # Get zone-specific factor
                        factor = zone_factor
                        
                        # Create base density prediction
                        if hasattr(self, 'target_stats') and self.output_size > 1:
                            # Use statistics from training data if available
                            base_density = predictions[i][0] * factor
                            # Scale to reasonable range using stored statistics
                            min_val = self.target_stats["min"][0] if isinstance(self.target_stats["min"], np.ndarray) else self.target_stats["min"]
                            max_val = self.target_stats["max"][0] if isinstance(self.target_stats["max"], np.ndarray) else self.target_stats["max"]
                            density = min_val + base_density * (max_val - min_val)
                            
                            # Create multi-metric prediction
                            prediction_dict[zone_id] = {
                                'density': float(density),
                                'speed': float(max(5.0, 60.0 - (density * 40))),  # Speed decreases as density increases
                                'flow_rate': float(density * max(5.0, 60.0 - (density * 40)) * 0.1)  # flow = density * speed * constant
                            }
                        else:
                            # Simplified single-output case
                            base_value = float(predictions[i][0] if len(predictions[i]) > 0 else predictions[i])
                            prediction_dict[zone_id] = base_value * factor
                    
                    predictions_list.append(prediction_dict)
                
                return predictions_list
            
            else:  # Single sequence or other array
                # Try to handle it as a single sequence
                try:
                    # Use the last time step with more weight
                    if len(X.shape) == 2:
                        last_values = X[-1, :]
                    else:
                        last_values = X
                    
                    base_prediction = np.mean(last_values)
                    
                    # Apply time factors
                    time_factor = 1.0
                    if hour >= morning_rush_start and hour <= morning_rush_end:
                        time_factor = 1.5
                    elif hour >= evening_rush_start and hour <= evening_rush_end:
                        time_factor = 1.6
                    
                    if is_weekend:
                        time_factor *= weekend_factor
                    
                    # Create prediction dictionary
                    prediction_dict = {}
                    for zone_id, zone_factor in zone_factors.items():
                        factor = zone_factor
                        base_value = float(base_prediction * factor * time_factor)
                        
                        if self.output_size > 1:
                            # Multi-output case
                            density = base_value
                            prediction_dict[zone_id] = {
                                'density': density,
                                'speed': max(5.0, 60.0 - (density * 40)),
                                'flow_rate': density * max(5.0, 60.0 - (density * 40)) * 0.1
                            }
                        else:
                            # Single output case
                            prediction_dict[zone_id] = base_value
                    
                    return [prediction_dict]
                
                except Exception as e:
                    print(f"Error in simple prediction: {e}")
                    # Fallback
                    return self._generate_fallback_prediction()
        
        # Fallback for non-array inputs or errors
        return self._generate_fallback_prediction()
    
    def _generate_fallback_prediction(self):
        """Generate a fallback prediction when other methods fail"""
        # Get current time for time-based patterns
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        is_weekend = day_of_week >= 5
        
        # Base traffic level depends on time of day
        if hour >= 7 and hour <= 9:  # Morning rush
            base_level = 0.7
        elif hour >= 16 and hour <= 18:  # Evening rush
            base_level = 0.8
        elif hour >= 10 and hour <= 15:  # Midday
            base_level = 0.5
        elif hour >= 19 and hour <= 22:  # Evening
            base_level = 0.4
        else:  # Night
            base_level = 0.2
        
        # Weekend adjustment
        if is_weekend:
            base_level *= 0.7
        
        # Zone factors from config
        zone_factors = self.config.get("zone_factors", {
            "zone_1": 1.0,
            "zone_2": 1.1,
            "zone_3": 0.9,
            "zone_4": 1.2
        })
        
        # Create prediction dictionary
        prediction_dict = {}
        
        for zone_id, zone_factor in zone_factors.items():
            density = base_level * zone_factor
            
            if self.output_size > 1:
                # Multi-output prediction
                prediction_dict[zone_id] = {
                    'density': float(density),
                    'speed': float(max(5.0, 60.0 - (density * 50))),
                    'flow_rate': float(density * max(5.0, 60.0 - (density * 50)) * 0.1)
                }
            else:
                # Single output prediction
                prediction_dict[zone_id] = float(density)
        
        return [prediction_dict]
    
    def save_model(self, path):
        """Save the model (dummy function)"""
        print(f"Simple model would be saved to {path}")
        
        # Save configuration and any learned statistics
        if hasattr(self, 'data_stats'):
            stats_path = path + ".stats.json"
            stats = {
                "data_stats": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.data_stats.items()},
                "config": self.config
            }
            
            if hasattr(self, 'target_stats'):
                stats["target_stats"] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.target_stats.items()}
            
            try:
                os.makedirs(os.path.dirname(stats_path), exist_ok=True)
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=4)
                print(f"Model statistics saved to {stats_path}")
            except Exception as e:
                print(f"Could not save statistics: {e}")
    
    def load_model(self, path):
        """Load model (dummy function)"""
        print(f"Simple model would be loaded from {path}")
        
        # Try to load statistics if available
        stats_path = path + ".stats.json"
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                
                if "data_stats" in stats:
                    self.data_stats = stats["data_stats"]
                if "target_stats" in stats:
                    self.target_stats = stats["target_stats"]
                if "config" in stats:
                    self.config = stats["config"]
                
                print(f"Loaded model statistics from {stats_path}")
            except Exception as e:
                print(f"Could not load statistics: {e}")