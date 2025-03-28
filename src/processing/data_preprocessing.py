import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import json
from datetime import datetime

class TrafficDataProcessor:
    """Process traffic data for model training and prediction"""
    
    def __init__(self):
        """Initialize the data processor"""
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        self.config_path = 'config/preprocessing_config.json'
        self.load_config()
    
    def load_config(self):
        """Load preprocessing configuration"""
        default_config = {
            "feature_selection": {
                "use_vehicle_count": True,
                "use_time_features": True,
                "use_zone_features": True,
                "use_speed_features": True,
                "use_flow_features": True
            },
            "sequence_params": {
                "default_time_steps": 10,
                "max_sequence_length": 50
            },
            "target_columns": ["traffic_density", "average_speed", "flow_rate"]
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Load config if exists, otherwise create default
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                print(f"Loaded preprocessing configuration from {self.config_path}")
            except:
                self.config = default_config
                print(f"Error loading config, using defaults")
        else:
            self.config = default_config
            # Save default config
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Created default preprocessing configuration at {self.config_path}")
    
    def prepare_features(self, traffic_data, time_steps=None, multi_output=True):
        """Prepare features for the traffic prediction model
        
        Args:
            traffic_data: DataFrame with traffic data
            time_steps: Number of time steps to use for sequence features
            multi_output: Whether to predict multiple outputs (density, speed, flow)
            
        Returns:
            X: Feature matrix
            y: Target values
        """
        # Use default time steps from config if not specified
        if time_steps is None:
            time_steps = self.config["sequence_params"]["default_time_steps"]
        
        # Ensure traffic_data is a DataFrame
        if not isinstance(traffic_data, pd.DataFrame):
            raise ValueError("traffic_data must be a pandas DataFrame")
        
        print(f"Preparing features from {len(traffic_data)} records with {time_steps} time steps")
        
        # Handle missing columns by checking for required columns
        required_columns = ['zone_id', 'vehicle_count']
        for col in required_columns:
            if col not in traffic_data.columns:
                if col == 'zone_id':
                    # Default to 'all' if zone_id is missing
                    traffic_data['zone_id'] = 'all'
                elif col == 'vehicle_count':
                    # Try to derive from related columns or default to 0
                    if any(col.startswith('zone_') for col in traffic_data.columns):
                        # Sum all zone counts if available
                        zone_cols = [col for col in traffic_data.columns if col.startswith('zone_')]
                        traffic_data['vehicle_count'] = traffic_data[zone_cols].sum(axis=1)
                    else:
                        traffic_data['vehicle_count'] = 0
                        print(f"Warning: 'vehicle_count' column missing and could not be derived")
        
        # Add time features if not present
        if 'hour' not in traffic_data.columns and 'timestamp' in traffic_data.columns:
            # Extract hour from timestamp
            try:
                traffic_data['timestamp'] = pd.to_datetime(traffic_data['timestamp'])
                traffic_data['hour'] = traffic_data['timestamp'].dt.hour
                traffic_data['day_of_week'] = traffic_data['timestamp'].dt.dayofweek
            except:
                print("Warning: Could not extract time features from timestamp")
        
        # If hour is still missing, add a default
        if 'hour' not in traffic_data.columns:
            traffic_data['hour'] = 12  # Default to noon
        
        # If day_of_week is missing, add a default
        if 'day_of_week' not in traffic_data.columns:
            traffic_data['day_of_week'] = 0  # Default to Monday
        
        # Extract relevant features
        feature_cols = []
        
        # Always include vehicle count
        feature_cols.append('vehicle_count')
        
        # Add zone_id if present
        if 'zone_id' in traffic_data.columns:
            # Convert to string to ensure it's hashable
            traffic_data['zone_id'] = traffic_data['zone_id'].astype(str)
            feature_cols.append('zone_id')
        
        # Add time features
        if self.config["feature_selection"]["use_time_features"]:
            if 'hour' in traffic_data.columns:
                feature_cols.append('hour')
            if 'day_of_week' in traffic_data.columns:
                feature_cols.append('day_of_week')
        
        # Add speed features if available and enabled
        if self.config["feature_selection"]["use_speed_features"]:
            speed_cols = [col for col in traffic_data.columns if 'speed' in col.lower()]
            feature_cols.extend(speed_cols)
        
        # Add flow features if available and enabled
        if self.config["feature_selection"]["use_flow_features"]:
            flow_cols = [col for col in traffic_data.columns if 'flow' in col.lower()]
            feature_cols.extend(flow_cols)
        
        # Select features
        features = traffic_data[feature_cols].copy()
        
        # One-hot encode categorical features
        categorical_cols = ['zone_id', 'day_of_week']
        for col in categorical_cols:
            if col in features.columns:
                features = pd.get_dummies(features, columns=[col])
        
        # Create cyclical features for time
        if 'hour' in features.columns:
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features = features.drop('hour', axis=1)
        
        # Save feature columns for later use
        self.feature_columns = features.columns.tolist()
        
        # Determine target columns based on multi_output flag
        if multi_output:
            target_columns = self.config["target_columns"]
            
            # Check if target columns exist, if not create synthetic ones
            for col in target_columns:
                if col not in traffic_data.columns:
                    print(f"Warning: Target column '{col}' not found, creating synthetic data")
                    if col == 'traffic_density':
                        # Derive from vehicle count
                        traffic_data[col] = traffic_data['vehicle_count'] / 100
                    elif col == 'average_speed':
                        # Create synthetic speed data
                        if 'traffic_density' in traffic_data.columns:
                            traffic_data[col] = 60 - (traffic_data['traffic_density'] * 40) + np.random.normal(0, 5, size=len(traffic_data))
                        else:
                            traffic_data[col] = np.random.uniform(10, 60, size=len(traffic_data))
                    elif col == 'flow_rate':
                        # Flow = density * speed
                        if 'traffic_density' in traffic_data.columns and 'average_speed' in traffic_data.columns:
                            traffic_data[col] = traffic_data['traffic_density'] * traffic_data['average_speed'] * 0.8
                        else:
                            traffic_data[col] = np.random.uniform(5, 50, size=len(traffic_data))
            
            # Extract targets
            y_data = traffic_data[target_columns].values
        else:
            # Single output - use vehicle_count as default target
            if 'traffic_density' in traffic_data.columns:
                y_data = traffic_data['traffic_density'].values
            else:
                y_data = traffic_data['vehicle_count'].values / 100  # Normalize to 0-1 range
        
        # Scale numerical features
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences for time series prediction
        X, y = self._create_sequences(scaled_features, y_data, time_steps)
        
        print(f"Prepared {X.shape[0]} sequences with shape {X.shape[1:]}, target shape: {y.shape}")
        
        return X, y
    
    def _create_sequences(self, feature_data, target_data, time_steps):
        """Create sequences for time series prediction
        
        Args:
            feature_data: Scaled feature matrix
            target_data: Target values
            time_steps: Number of time steps in each sequence
            
        Returns:
            X: Sequence features
            y: Target values (next time step)
        """
        X, y = [], []
        
        # Handle multi-dimensional targets
        is_multi_target = len(target_data.shape) > 1
        
        for i in range(len(feature_data) - time_steps):
            # Add sequence to features
            X.append(feature_data[i:i + time_steps])
            
            # Add target (next time step)
            if is_multi_target:
                y.append(target_data[i + time_steps])
            else:
                y.append(target_data[i + time_steps])
        
        return np.array(X), np.array(y)
    
    def preprocess_new_data(self, new_data, time_steps=None, historical_data=None):
        """Preprocess new data for prediction
        
        Args:
            new_data: New traffic data (dictionary or DataFrame)
            time_steps: Number of time steps in sequence
            historical_data: Optional historical data to use for sequence
            
        Returns:
            Preprocessed data ready for model prediction
        """
        # Use default time steps from config if not specified
        if time_steps is None:
            time_steps = self.config["sequence_params"]["default_time_steps"]
        
        try:
            # Convert the traffic state to a DataFrame
            if isinstance(new_data, dict):
                # Handle simulation state format
                if 'vehicles' in new_data:
                    # Extract relevant data from simulation state
                    df_data = {
                        'hour': [new_data.get('hour', datetime.now().hour)],
                        'day_of_week': [new_data.get('day_of_week', datetime.now().weekday())]
                    }
                    
                    # Add vehicle counts for each zone
                    vehicles = new_data.get('vehicles', {})
                    total_vehicles = 0
                    
                    for zone_id, count in vehicles.items():
                        # Convert zone_id to string to ensure it's hashable
                        zone_id_str = str(zone_id)
                        df_data[f'zone_{zone_id_str}'] = [count]
                        total_vehicles += count
                    
                    # Add total vehicle count
                    df_data['vehicle_count'] = [total_vehicles]
                    
                    # Create DataFrame
                    features = pd.DataFrame(df_data)
                    
                    # Add zone_id column for compatibility (using a dummy value)
                    features['zone_id'] = 'all'
                else:
                    # Try to convert flat dictionary to DataFrame
                    try:
                        # Convert to list of tuples for DataFrame constructor
                        items = [(k, v) for k, v in new_data.items()]
                        features = pd.DataFrame([dict(items)])
                    except:
                        print("Error: Could not convert traffic state to DataFrame")
                        return self._create_dummy_sequence(time_steps)
            else:
                # If already a DataFrame, use it directly
                features = pd.DataFrame(new_data)
            
            # Add current time features if missing
            if 'hour' not in features.columns:
                features['hour'] = datetime.now().hour
            if 'day_of_week' not in features.columns:
                features['day_of_week'] = datetime.now().weekday()
            
            # Create cyclical time features
            if 'hour' in features.columns:
                features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
                features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
                features = features.drop('hour', axis=1)
            
            # Handle categorical features safely
            if 'zone_id' in features.columns:
                # Convert to string to ensure hashability
                features['zone_id'] = features['zone_id'].astype(str)
                features = pd.get_dummies(features, columns=['zone_id'])
            
            if 'day_of_week' in features.columns:
                # Convert to string to ensure hashability
                features['day_of_week'] = features['day_of_week'].astype(str)
                features = pd.get_dummies(features, columns=['day_of_week'])
            
            # Ensure all expected feature columns are present
            if self.feature_columns is not None:
                for col in self.feature_columns:
                    if col not in features.columns:
                        features[col] = 0
                
                # Select only columns used in training
                features = features[self.feature_columns]
            
            # Scale features
            if hasattr(self, 'scaler') and self.scaler is not None:
                try:
                    scaled_features = self.scaler.transform(features)
                except:
                    # If scaling fails, use simple normalization
                    print("Warning: Scaler transform failed, using simple normalization")
                    scaled_features = (features - features.mean()) / features.std().fillna(1)
                    scaled_features = scaled_features.fillna(0).values
            else:
                # No scaler available, use values directly
                scaled_features = features.values
            
            # Create sequence using historical data if provided
            if historical_data is not None and len(historical_data) >= time_steps:
                # Use the most recent historical data
                sequence = historical_data[-time_steps:].reshape(1, time_steps, historical_data.shape[1])
                return sequence
            
            # If we have enough data points in the current input
            elif len(scaled_features) >= time_steps:
                sequence = scaled_features[-time_steps:].reshape(1, time_steps, scaled_features.shape[1])
                return sequence
            else:
                # For initial prediction when we don't have enough history,
                # repeat the current state to fill the sequence
                current_state = scaled_features[-1:]
                repeated_state = np.repeat(current_state, time_steps, axis=0)
                sequence = repeated_state.reshape(1, time_steps, scaled_features.shape[1])
                return sequence
        
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return self._create_dummy_sequence(time_steps)
    
    def _create_dummy_sequence(self, time_steps):
        """Create a dummy sequence when preprocessing fails"""
        # Determine feature count from feature_columns or use default
        dummy_features = 5  # Default number of features
        if self.feature_columns is not None:
            dummy_features = len(self.feature_columns)
        
        # Create a sequence of zeros with proper shape
        dummy_sequence = np.zeros((1, time_steps, dummy_features))
        print("Warning: Created dummy sequence due to preprocessing error")
        
        return dummy_sequence
    
    def save_processor_state(self, path):
        """Save the processor state for later use"""
        state = {
            "feature_columns": self.feature_columns,
            "config": self.config
        }
        
        # Save scaler if available
        if hasattr(self, 'scaler') and self.scaler is not None:
            from sklearn.externals import joblib
            scaler_path = os.path.join(os.path.dirname(path), "scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            state["scaler_path"] = scaler_path
        
        # Save state
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=4)
        
        print(f"Processor state saved to {path}")
    
    def load_processor_state(self, path):
        """Load processor state from file"""
        if not os.path.exists(path):
            print(f"Warning: No processor state found at {path}")
            return False
        
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            self.feature_columns = state.get("feature_columns")
            self.config = state.get("config", self.config)
            
            # Load scaler if available
            if "scaler_path" in state:
                from sklearn.externals import joblib
                scaler_path = state["scaler_path"]
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
            
            print(f"Processor state loaded from {path}")
            return True
        
        except Exception as e:
            print(f"Error loading processor state: {e}")
            return False
