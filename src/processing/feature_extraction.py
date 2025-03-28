import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

class TrafficFeatureExtractor:
    """Extract features from traffic data for model input"""
    
    def __init__(self, config_path='config/feature_extraction_config.json'):
        """Initialize the feature extractor"""
        self.config_path = config_path
        self.load_config()
    
    def load_config(self):
        """Load feature extraction configuration"""
        default_config = {
            "time_features": {
                "use_hour": True,
                "use_minute": True,
                "use_day_of_week": True,
                "use_month": True,
                "use_cyclical_encoding": True
            },
            "spatial_features": {
                "use_zone_id": True,
                "use_direction": True,
                "use_intersection": True,
                "use_coordinates": False
            },
            "traffic_features": {
                "use_vehicle_count": True,
                "use_vehicle_types": True,
                "use_speed_data": True,
                "use_density": True
            },
            "rush_hours": {
                "morning_start": 7,
                "morning_end": 9,
                "evening_start": 16,
                "evening_end": 18
            }
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Load config if exists, otherwise create default
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                print(f"Loaded feature extraction configuration from {self.config_path}")
            except:
                self.config = default_config
                print(f"Error loading config, using defaults")
        else:
            self.config = default_config
            # Save default config
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Created default feature extraction configuration at {self.config_path}")
    
    def extract_time_features(self, timestamp=None):
        """Extract time-based features from timestamp
        
        Args:
            timestamp: Datetime object or timestamp string (defaults to current time)
            
        Returns:
            Dictionary of time features
        """
        # Use current time if no timestamp provided
        if timestamp is None:
            dt = datetime.now()
        elif isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                try:
                    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                except:
                    print(f"Warning: Could not parse timestamp '{timestamp}', using current time")
                    dt = datetime.now()
        elif isinstance(timestamp, datetime):
            dt = timestamp
        else:
            raise ValueError("timestamp must be a datetime object or ISO format string")
        
        # Extract basic time features
        features = {}
        
        time_config = self.config["time_features"]
        rush_config = self.config["rush_hours"]
        
        if time_config["use_hour"]:
            features['hour'] = dt.hour
        
        if time_config["use_minute"]:
            features['minute'] = dt.minute
        
        if time_config["use_day_of_week"]:
            features['day_of_week'] = dt.weekday()
        
        if time_config["use_month"]:
            features['month'] = dt.month
        
        # Add derived time features
        features['is_weekend'] = 1 if dt.weekday() >= 5 else 0
        
        # Check if current time is during rush hour
        is_morning_rush = (dt.hour >= rush_config["morning_start"] and 
                           dt.hour <= rush_config["morning_end"])
        is_evening_rush = (dt.hour >= rush_config["evening_start"] and 
                           dt.hour <= rush_config["evening_end"])
        features['is_rush_hour'] = 1 if (is_morning_rush or is_evening_rush) else 0
        
        # Add specific rush hour indicators
        features['is_morning_rush'] = 1 if is_morning_rush else 0
        features['is_evening_rush'] = 1 if is_evening_rush else 0
        
        # Add cyclical encoding for time features
        if time_config["use_cyclical_encoding"]:
            # Hour of day (0-23)
            features['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
            
            # Day of week (0-6)
            features['day_sin'] = np.sin(2 * np.pi * dt.weekday() / 7)
            features['day_cos'] = np.cos(2 * np.pi * dt.weekday() / 7)
            
            # Month of year (1-12)
            features['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
        
        return features
    
    def extract_spatial_features(self, zone_id, zone_data=None):
        """Extract spatial features for a traffic zone
        
        Args:
            zone_id: ID of the traffic zone
            zone_data: Data about the zone (coordinates, direction, etc.)
            
        Returns:
            Dictionary of spatial features
        """
        features = {'zone_id': zone_id}
        
        # If no zone data provided, return just the zone_id
        if zone_data is None:
            return features
        
        spatial_config = self.config["spatial_features"]
        
        # Extract direction if available and enabled
        if spatial_config["use_direction"] and 'direction' in zone_data:
            features['direction'] = zone_data['direction']
        
        # Extract intersection status if available and enabled
        if spatial_config["use_intersection"]:
            features['is_intersection'] = 1 if zone_data.get('is_intersection', False) else 0
        
        # Extract coordinates if available and enabled
        if spatial_config["use_coordinates"]:
            if 'coordinates' in zone_data:
                coords = zone_data['coordinates']
                if isinstance(coords, dict):
                    features['latitude'] = coords.get('lat', 0)
                    features['longitude'] = coords.get('lng', 0)
                elif isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    features['latitude'] = coords[0]
                    features['longitude'] = coords[1]
        
        # Extract road type if available
        if 'road_type' in zone_data:
            features['road_type'] = zone_data['road_type']
        
        # Extract speed limit if available
        if 'speed_limit' in zone_data:
            features['speed_limit'] = zone_data['speed_limit']
        
        return features
    
    def extract_traffic_features(self, vehicle_counts, speed_data=None, density_data=None):
        """Extract traffic-related features
        
        Args:
            vehicle_counts: Dictionary of vehicle counts by type
            speed_data: Optional speed data
            density_data: Optional density data
            
        Returns:
            Dictionary of traffic features
        """
        traffic_config = self.config["traffic_features"]
        features = {}
        
        # Process vehicle counts
        if isinstance(vehicle_counts, dict):
            total_count = sum(vehicle_counts.values())
        elif isinstance(vehicle_counts, (int, float)):
            total_count = vehicle_counts
            vehicle_counts = {'total': vehicle_counts}
        else:
            print(f"Warning: Unexpected vehicle_counts format: {type(vehicle_counts)}")
            total_count = 0
            vehicle_counts = {}
        
        # Add total vehicle count if enabled
        if traffic_config["use_vehicle_count"]:
            features['vehicle_count'] = total_count
        
        # Add vehicle type ratios if enabled
        if traffic_config["use_vehicle_types"] and total_count > 0:
            # Calculate ratios for different vehicle types
            for vehicle_type in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                if vehicle_type in vehicle_counts:
                    features[f'{vehicle_type}_ratio'] = vehicle_counts[vehicle_type] / max(total_count, 1)
                    features[f'{vehicle_type}_count'] = vehicle_counts[vehicle_type]
        
        # Add speed data if available and enabled
        if traffic_config["use_speed_data"] and speed_data:
            if isinstance(speed_data, dict):
                for key, value in speed_data.items():
                    features[f'speed_{key}'] = value
            else:
                # If speed_data is not a dict, assume it's a single value
                features['avg_speed'] = float(speed_data)
        
        # Add density data if available and enabled
        if traffic_config["use_density"] and density_data:
            if isinstance(density_data, dict):
                for key, value in density_data.items():
                    features[f'density_{key}'] = value
            else:
                # If density_data is not a dict, assume it's a single value
                features['traffic_density'] = float(density_data)
        
        # Calculate derived metrics
        if 'vehicle_count' in features and 'avg_speed' in features:
            # Calculate flow rate (vehicles per hour) as density * speed
            # This is a simplified calculation
            features['flow_rate'] = features['vehicle_count'] * features['avg_speed'] * 0.1
        
        # Calculate congestion level (0-1 scale)
        if 'traffic_density' in features:
            features['congestion_level'] = min(1.0, features['traffic_density'] / 0.8)
        elif 'vehicle_count' in features:
            # Estimate congestion based on vehicle count
            # This is very simplified and should be calibrated
            features['congestion_level'] = min(1.0, features['vehicle_count'] / 100)
        
        return features
    
    def combine_features(self, time_features, spatial_features, traffic_features):
        """Combine all features into a single feature vector
        
        Args:
            time_features: Dictionary of time features
            spatial_features: Dictionary of spatial features
            traffic_features: Dictionary of traffic features
            
        Returns:
            Combined feature dictionary
        """
        combined = {}
        
        # Add time features
        combined.update(time_features)
        
        # Add spatial features
        combined.update(spatial_features)
        
        # Add traffic features
        combined.update(traffic_features)
        
        return combined
    
    def extract_all_features(self, data):
        """Extract all features from a data dictionary
        
        Args:
            data: Dictionary containing timestamp, zone_id, vehicle_counts, etc.
            
        Returns:
            Dictionary of all extracted features
        """
        # Extract timestamp
        timestamp = data.get('timestamp', datetime.now())
        time_features = self.extract_time_features(timestamp)
        
        # Extract zone information
        zone_id = data.get('zone_id', 'unknown')
        zone_data = data.get('zone_data', {})
        spatial_features = self.extract_spatial_features(zone_id, zone_data)
        
        # Extract traffic information
        vehicle_counts = data.get('vehicle_counts', {})
        speed_data = data.get('speed_data', None)
        density_data = data.get('density_data', None)
        traffic_features = self.extract_traffic_features(vehicle_counts, speed_data, density_data)
        
        # Combine all features
        return self.combine_features(time_features, spatial_features, traffic_features)
    
    def process_batch(self, data_batch):
        """Process a batch of data points
        
        Args:
            data_batch: List of data dictionaries
            
        Returns:
            DataFrame with extracted features for all data points
        """
        all_features = []
        
        for data_point in data_batch:
            features = self.extract_all_features(data_point)
            all_features.append(features)
        
        # Convert to DataFrame
        return pd.DataFrame(all_features)